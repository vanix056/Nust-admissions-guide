import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_nust_faqs(url):
    print(f"Scraping: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        faq_data = []

        
        cards = soup.find_all('div', class_='card')
        
        # Loop through each card to extract question and answer
        for card in cards:
            q_button = card.find('button', class_='btn-link')
            a_div = card.find('div', class_='card-body')
            
            if q_button and a_div:
                # Clean up the text (removing extra whitespace/newlines)
                question = q_button.get_text(strip=True)
                answer = a_div.get_text(strip=True)
                
                faq_data.append({
                    "question": question,
                    "answer": answer
                })
                print(f"  [Found]: {question[:50]}...")

        return faq_data

    except Exception as e:
        print(f"Error on {url}: {e}")
        return []

# URLs to target
urls = [
    "https://nust.edu.pk/faqs/",
    "https://nust.edu.pk/faq-category/ug-admission/",
    "https://nust.edu.pk/faq-category/mbbs-admissions-faqs/",
    "https://nust.edu.pk/faq-category/bshnd-admissions-faqs/"
]

all_results = []
for link in urls:
    data = scrape_nust_faqs(link)
    print(f"Total pairs found on this page: {len(data)}")
    all_results.extend(data)
    time.sleep(1)

# Save to file
filename = 'nust_faq.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print(f"\nSuccess! Total FAQs collected: {len(all_results)}")
print(f"File saved as: {filename}")