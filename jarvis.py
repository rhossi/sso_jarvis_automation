import asyncio
import os
from playwright.async_api import async_playwright
import pandas as pd
import json
import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage

load_dotenv()

async def login_and_save_session(target_url, auth_state_file="auth_state.json"):
    """
    Perform initial login and save session state.
    Run this once to authenticate and save session.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print("Opening login page...")
        await page.goto(target_url)
        
        print("\n" + "="*60)
        print("PLEASE COMPLETE SSO LOGIN IN THE BROWSER")
        print("="*60)
        print("1. Complete your SSO authentication")
        print("2. Make sure you can see the target application")
        print("3. Press Enter here when login is complete")
        print("="*60)
        
        input("Press Enter after completing login...")
        
        # Save the authentication state
        await context.storage_state(path=auth_state_file)
        print(f"Authentication state saved to {auth_state_file}")
        
        await browser.close()

async def run_automation_with_saved_session(target_url, search_term, auth_state_file="auth_state.json"):
    """
    Run automation using saved authentication state.
    """
    if not os.path.exists(auth_state_file):
        print(f"Auth state file {auth_state_file} not found!")
        print("Please run login_and_save_session() first")
        return None
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        
        # Load the saved authentication state
        context = await browser.new_context(storage_state=auth_state_file)
        page = await context.new_page()
        
        try:
            # Step 1: Open the URL
            print(f"Opening application with saved session...")
            await page.goto(target_url, wait_until="networkidle")
            
            # Check if we're still authenticated
            if "login" in page.url.lower() or "auth" in page.url.lower():
                print("Session expired! Need to re-authenticate.")
                await browser.close()
                return None
            
            print("Session is valid!")
            
            # Step 2: Find the href and follow it
            print("Looking for Opportunity Engagement link...")
            opportunity_link = page.locator('xpath=//li[.//h3[text()="Opportunity Engagement"]]//a')
            await opportunity_link.wait_for(timeout=30000)
            
            href_value = await opportunity_link.get_attribute("href")
            print(f"Found Opportunity Engagement link: {href_value}")
            
            # Navigate to the Opportunity Engagement page
            await opportunity_link.click()
            await page.wait_for_load_state("networkidle")
            print("Successfully navigated to Opportunity Engagement page")
            
            # Step 3: Find the text box and fill in the provided value
            print("Looking for search field...")
            search_field = page.locator("#R1934169132892269954_fr_search")
            await search_field.wait_for(timeout=30000)
            
            print(f"Filling search field with: {search_term}")
            await search_field.clear()
            await search_field.fill(search_term)
            
            # Step 4: Hit enter
            print("Hitting Enter to search...")
            await search_field.press("Enter")
            
            # Wait for results to load
            await page.wait_for_timeout(3000)
            
            # Step 5: Extract data from the table
            print("Extracting table data...")
            table = page.locator("#report_table_my_report")
            await table.wait_for(timeout=30000)
            
            # Get table data
            headers = await table.locator("tr").first.locator("th, td").all_text_contents()
            rows = table.locator("tr")
            row_count = await rows.count()
            
            data = []
            for i in range(1, row_count):  # Skip header
                row = rows.nth(i)
                cells = await row.locator("td").all_text_contents()
                
                if cells:
                    row_data = {}
                    for j, cell_text in enumerate(cells):
                        header = headers[j] if j < len(headers) else f"Column_{j+1}"
                        row_data[header] = cell_text.strip()
                    data.append(row_data)
            
            print(f"Extracted {len(data)} rows")
            
            # Save data
            if data:
                df = pd.DataFrame(data)
                df.to_csv("extracted_data.csv", index=False)
                print("Data saved to extracted_data.csv")
            
            return data
            
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            await browser.close()

def generate_llm_summary_oci(table_data):
    COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
    AUTH_TYPE = "API_KEY"
    CONFIG_PROFILE = "DEFAULT"

    # Prepare data for summarization
    data_summary = {
        "total_rows": len(table_data),
        "columns": list(table_data[0].keys()) if table_data else [],
        "sample_data": table_data[:5] if len(table_data) > 5 else table_data
    }

    # Create prompt for Grok
    prompt = f"""
        Please provide a concise summary of this table data:

        Total rows: {data_summary['total_rows']}
        Columns: {', '.join(data_summary['columns'])}

        Sample data (first 5 rows):
        {json.dumps(data_summary['sample_data'], indent=2)}

        Please summarize:
        1. What type of data this appears to be
        2. Key patterns or insights you notice
        3. Any notable trends or observations
        4. A brief overview of the main findings

        Keep the summary concise and actionable.
    """

    # Service endpoint
    endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

    # initialize interface
    chat = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=endpoint,
        compartment_id=COMPARTMENT_ID,
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE
    )

    messages = [
        HumanMessage(content=prompt),
    ]

    response = chat.invoke(messages)

    return response.content


    url = os.getenv("OLLAMA_URL")
    
        # Prepare data for summarization
    data_summary = {
        "total_rows": len(table_data),
        "columns": list(table_data[0].keys()) if table_data else [],
        "sample_data": table_data[:5] if len(table_data) > 5 else table_data
    }
    
    # Create prompt for Grok
    prompt = f"""
        Please provide a concise summary of this table data:

        Total rows: {data_summary['total_rows']}
        Columns: {', '.join(data_summary['columns'])}

        Sample data (first 5 rows):
        {json.dumps(data_summary['sample_data'], indent=2)}

        Please summarize:
        1. What type of data this appears to be
        2. Key patterns or insights you notice
        3. Any notable trends or observations
        4. A brief overview of the main findings

        Keep the summary concise and actionable.
    """
    
    data = {
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a data analyst assistant. Provide clear, concise summaries of tabular data."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "stream": True
    }
    
    response = requests.post(url, json=data, stream=True)
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if not chunk.get('done'):
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                else:
                    print()  # New line when done
    else:
        print(f"Error: {response.status_code}")

async def main():
    """Main function - handles both login and automation."""
    target_url = "https://itfcuqba1dqacqh-dtcinnovate.adb.us-ashburn-1.oraclecloudapps.com/ords/r/dtc_oci_innovation/customer-engagement-retrospective/engagement-analysis-report?session=1101789845951936"
    search_term = input("What is your search term?")  # Replace with your search term
    auth_file = "oracle_sso_session.json"
    
    # Check if we have a saved session
    if not os.path.exists(auth_file):
        print("No saved session found. Starting initial login...")
        await login_and_save_session(target_url, auth_file)
        print("\nSession saved! Now running automation...")
    
    # Run the automation
    data = await run_automation_with_saved_session(target_url, search_term, auth_file)
    
    if data:
        print(f"Success! Extracted {len(data)} rows")
        
        # Show first few rows
        print("\nSample data:")
        for i, row in enumerate(data[:3]):
            print(f"Row {i+1}: {row}")
        
        # Call Grok for summary
        print("\n" + "="*60)
        print("GENERATING AI SUMMARY WITH LLM")
        print("="*60)
        
        # summary = generate_llm_summary(data)
        summary = generate_llm_summary_oci(data)
        print("\nGROK SUMMARY:")
        print("-" * 40)
        print(summary)
        print("-" * 40)
        
        # Save summary along with data
        output_data = {
            "extraction_info": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "search_term": search_term,
                "total_rows": len(data)
            },
            "grok_summary": summary,
            "table_data": data
        }
        
        # Save comprehensive output
        with open("complete_analysis.json", "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nComplete analysis saved to complete_analysis.json")
        
    else:
        print("Failed to extract data. You may need to re-authenticate.")
        # Delete the auth file to force re-login next time
        if os.path.exists(auth_file):
            os.remove(auth_file)
            print("Cleared saved session. Run the script again to re-authenticate.")

if __name__ == "__main__":
    asyncio.run(main())