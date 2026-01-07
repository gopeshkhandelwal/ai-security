#!/usr/bin/env python3
"""Download all public snippets from GitLab instance"""

import os
import requests
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

GITLAB_URL = "https://gitlab.practical-devsecops.training"
SNIPPETS_DIR = Path.home() / "Code/CAISP/snippets"

def get_all_snippets():
    """Scrape all public snippets from explore page"""
    snippets = []
    page = 1
    
    while True:
        print(f"Fetching page {page}...")
        url = f"{GITLAB_URL}/explore/snippets"
        params = {"page": page}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all snippet links
            snippet_links = soup.find_all('a', href=re.compile(r'/-/snippets/\d+$'))
            
            if not snippet_links:
                break
            
            for link in snippet_links:
                snippet_url = link.get('href')
                match = re.search(r'/-/snippets/(\d+)$', snippet_url)
                if match:
                    snippet_id = match.group(1)
                    title = link.get_text(strip=True)
                    
                    # Avoid duplicates
                    if not any(s['id'] == snippet_id for s in snippets):
                        snippets.append({
                            'id': snippet_id,
                            'title': title,
                            'url': f"{GITLAB_URL}{snippet_url}"
                        })
            
            print(f"  Found {len(snippet_links)} snippets on page {page}")
            
            # Check if there's a next page
            next_button = soup.find('a', rel='next')
            if not next_button:
                break
                
            page += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching snippets: {e}")
            break
    
    return snippets

def download_snippet_files(snippet_id, snippet_title, snippet_url):
    """Download all files from a snippet"""
    try:
        # Get snippet page
        response = requests.get(snippet_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Create directory for snippet
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_', '.') else '_' for c in snippet_title)
        snippet_dir = SNIPPETS_DIR / f"{snippet_id}_{safe_title}"
        snippet_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all file entries - look for elements with file paths
        file_elements = soup.find_all('div', class_=re.compile(r'file-holder|blob-viewer'))
        
        # Alternative: look for the filename in the page and construct raw URL
        # Pattern: https://gitlab.../snippets/{id}/raw/main/{filename}
        filename = snippet_title
        
        # Try to download using the raw URL pattern
        raw_url = f"{GITLAB_URL}/-/snippets/{snippet_id}/raw/main/{filename}"
        
        try:
            file_response = requests.get(raw_url, timeout=30)
            file_response.raise_for_status()
            
            output_file = snippet_dir / filename
            output_file.write_bytes(file_response.content)
            print(f"  ✓ Downloaded: {filename}")
            return True
        except requests.exceptions.RequestException:
            # If direct download fails, try to find file links in HTML
            pass
        
        # Fallback: search for any raw/download links in the page
        all_links = soup.find_all('a')
        downloaded = False
        
        for link in all_links:
            href = link.get('href', '')
            if '/raw/main/' in href or 'download' in href.lower():
                if not href.startswith('http'):
                    href = f"{GITLAB_URL}{href}"
                
                # Extract filename
                fname = href.split('/')[-1].split('?')[0]
                if not fname or fname == 'main':
                    fname = snippet_title
                
                try:
                    file_response = requests.get(href, timeout=30)
                    file_response.raise_for_status()
                    
                    output_file = snippet_dir / fname
                    output_file.write_bytes(file_response.content)
                    print(f"  ✓ Downloaded: {fname}")
                    downloaded = True
                except:
                    continue
        
        if not downloaded:
            print(f"  ✗ No files found")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading snippet {snippet_id}: {e}")
        return False

def main():
    print("=" * 60)
    print("GitLab Snippets Downloader")
    print("=" * 60)
    print(f"Target: {GITLAB_URL}")
    print(f"Output: {SNIPPETS_DIR}")
    print("=" * 60)
    
    # Create output directory
    SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all snippets
    snippets = get_all_snippets()
    print(f"\nTotal snippets found: {len(snippets)}")
    
    if not snippets:
        print("No snippets found or API not accessible")
        return
    
    # Download each snippet
    print("\nDownloading snippets...")
    success_count = 0
    
    for i, snippet in enumerate(snippets, 1):
        snippet_id = snippet['id']
        snippet_title = snippet.get('title', f'snippet_{snippet_id}')
        snippet_url = snippet['url']
        
        print(f"\n[{i}/{len(snippets)}] {snippet_title} (ID: {snippet_id})")
        
        if download_snippet_files(snippet_id, snippet_title, snippet_url):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✓ Downloaded {success_count}/{len(snippets)} snippets")
    print(f"Location: {SNIPPETS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
