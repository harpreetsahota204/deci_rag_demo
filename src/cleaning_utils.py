import os 
import re
from pathlib import Path
from typing import List, Tuple, Dict

import fitz

def extract_and_process_text(pdf_file: str) -> str:
    """
    Extracts and aggregates text from all pages of a given PDF file.

    This function opens a PDF file, iterates through each page, extracts the text,
    and concatenates it. It prints a message indicating the processing of the file.
    If an error occurs during processing, an error message is printed.

    Parameters:
    - pdf_file (str): The path to the PDF file to be processed.

    Returns:
    - str: The aggregated text extracted from the PDF file.

    Raises:
    - Exception: If any error occurs during the opening or processing of the PDF file.
    """
    text = ""
    try:
        with fitz.open(pdf_file) as doc:  # Open the PDF file
            for page in doc:  # Iterate through each page
                text += page.get_text()  # Extract and concatenate text
        print(f"Processing {pdf_file}...")  # Indicate processing of the file
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")  # Print any errors
    return text  # Return the aggregated text

def remove_copyright_notice(text: str) -> str:
    """
    Removes copyright notice from the provided text.

    This function searches for patterns that match copyright notices typically
    found in texts and removes them. The pattern looks for the copyright symbol
    followed by a year and any text that ends with 'Inc.' (optionally followed by
    a period), and removes this whole segment from the text.

    Parameters:
    - text (str): The text from which copyright notices are to be removed.

    Returns:
    - str: The text after removing any found copyright notices.
    """
    # Define the pattern for copyright notices
    pattern = r"©\s*\d{4}\s*.*?Inc\.?"
    # Remove the identified pattern from the text
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text

def remove_dot_sequences(text):
    """
    Removes sequences of dots that may represent ellipses or formatting artifacts,
    typically found in tables of contents or similar structured text. Spaces between
    dots in the sequence are also considered and removed.

    Parameters:
    - text (str): The input text from which dot sequences will be removed.

    Returns:
    - str: The cleaned text with dot sequences removed.
    """
    pattern = r"\.\s*(\.\s*)+"
    cleaned_text = re.sub(pattern, " ", text)
    
    return cleaned_text

def scrub_text(text: str) -> str:
    """
    Cleans the provided text by removing specific consolidated patterns and standardizing whitespace.

    This function uses consolidated regex patterns to remove phrases, trademarks,
    copyright symbols, specific company mentions, and other specified unwanted text
    from the input. It normalizes multiple consecutive spaces to a single space and
    trims leading and trailing whitespace.

    Parameters:
    - text (str): The text to be cleaned.

    Returns:
    - str: The cleaned text, with unwanted patterns removed and whitespace normalized.
    """
    # Consolidated patterns to remove
    patterns_to_remove = [
        # Combines case-insensitive matches and similar phrases
        (r"(All rights reserved|SOLUTION(S)? BRIEF|Executive Summary|TABLE OF CONTENTS|www\.supermicro\.com|Super\s+Micro\s+Computer,\s+Inc\.\s+980\s+Rock\s+Avenue\s+San\s+Jose,\s+CA\s+95131\s+USA)", re.IGNORECASE),
        # Specific mentions without case sensitivity or special symbols
        (r"(Copyright Super Micro Computer, Inc\.|SUPERMICRO|\(Nasdaq: SMCI\))", None),
        # Trademark, copyright, and registration symbols
        (r"[\®\™\©]", None),
    ]
    
    # Remove each pattern from the text
    for pattern, flag in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=flag) if flag else re.sub(pattern, "", text)
    
    # Additional cleanup for any resulting double spaces or leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_and_remove_all_dates(text):
    """
    Searches for all date patterns in the specified text, extracts the first found date,
    and removes all occurrences of date patterns from the text. It accommodates date patterns
    formatted as "[Month Name] [4-digit Year]" and "[Month Name], [4-digit Year]", and the search
    is case-insensitive.
    
    Args:
    - text (str): The input text from which date patterns need to be extracted and removed.
    
    Returns:
    - tuple: A tuple containing the first found date (str) and the cleaned text (str) with all date patterns removed.
             If no date is found, the date part of the tuple will be None.
    """
    
    # Adjust the regex pattern to include dates with and without a comma, and make it case-insensitive
    date_pattern = r"\b(january|february|march|april|may|june|july|august|september|october|november|december)[,]?\s+(\d{4})\b"
    
    # Compile the regex with the IGNORECASE flag to make it case-insensitive
    compiled_pattern = re.compile(date_pattern, re.IGNORECASE)
    
    # Find all occurrences of the date pattern in the text
    matches = compiled_pattern.findall(text)
    first_date = None
    
    if matches:
        # Format the first date as "Month Year" without a comma
        first_date = f"{matches[0][0].capitalize()} {matches[0][1]}"
    
    # Remove all occurrences of the date pattern from the text
    cleaned_text = compiled_pattern.sub("", text).strip()
    
    # Further cleanup to reduce potential multiple consecutive spaces or newlines caused by removals
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # Reduce multiple spaces to a single space
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Reduce multiple newlines to a single newline
    
    return first_date, cleaned_text

def extract_and_remove_all_websites(text):
    """
    Searches for all website patterns in the specified text, extracts all found websites,
    and removes all occurrences of website patterns from the text. The function is designed
    to capture a wide range of URLs.
    
    Args:
    - text (str): The input text from which website patterns need to be extracted and removed.
    
    Returns:
    - tuple: A tuple containing a list of all found websites and the cleaned text with all website patterns removed.
    """
    website_pattern = r"https?://[^\s]+|www\.[^\s]+"
    # Compile the regex with the IGNORECASE flag to make it case-insensitive
    compiled_pattern = re.compile(website_pattern, re.IGNORECASE)
    
    # Find all occurrences of the website pattern in the text
    matches = set(compiled_pattern.findall(text))
    
    # Remove all occurrences of the website pattern from the text
    cleaned_text = compiled_pattern.sub("", text).strip()
    
    # Further cleanup for spacing and newlines
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # Reduce multiple spaces to a single space
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Reduce multiple newlines to a single newline
    
    return list(matches), cleaned_text


def remove_text_after_phrases(text, proximity_ratio=0.25):
    """
    Removes all text after specified phrases towards the end of the document, case-insensitively.
    Only acts on the last part of the document defined by proximity_ratio.

    Args:
    - text (str): The input text from which to remove trailing sections after specific phrases.
    - proximity_ratio (float): Defines the portion at the end of the text to search for the phrases,
                               e.g., 0.25 means the last 25% of the text.

    Returns:
    - str: The cleaned text with specified trailing sections removed if they occur within the
           defined proximity to the end of the document.
    """

    # List of phrases to search for towards the end of the document
    phrases = [
        "for more information",
        "references",
        "The information contained in this document is subject to change without notice.",
        "learn more at",
        "to find out more about",
        "To learn more about",
        "additional resources",
        "trademarks",
        "For more certified",
        "ABOUT SUPER MICRO COMPUTER"
    ]

    # Determine the starting index for 'the end' of the document based on proximity_ratio
    start_index = int(len(text) * (1 - proximity_ratio))

    # Extract the end portion of the document to search within
    end_portion = text[start_index:]

    # Combine the phrases into a single regex pattern, case-insensitive
    pattern = r"(?:" + "|".join(re.escape(phrase) for phrase in phrases) + r").*"

    # Search for the pattern in the end portion of the document
    if re.search(pattern, end_portion, flags=re.IGNORECASE | re.DOTALL):
        # If found, remove all text from the first occurrence of any phrase to the end
        modified_end_portion = re.sub(pattern, "", end_portion, flags=re.IGNORECASE | re.DOTALL).strip()
        # Reconstruct the text with the modified end portion
        cleaned_text = text[:start_index] + modified_end_portion
    else:
        # If no phrases are found in the end portion, return the text as is
        cleaned_text = text

    return cleaned_text.strip()

def clean_and_prepare_texts(pdf_directory: str) -> List[Tuple[str, Dict]]:
    """
    Processes PDF files in a specified directory, applying a series of text cleaning operations to each,
    and collects metadata for each file. The function extracts text from each PDF file, cleans it using
    various utilities, and compiles metadata including the file name, publication date, and referenced websites.

    The cleaning operations include extracting and removing dates and websites, removing dot sequences and
    copyright notices, scrubbing the text, and removing specific phrases towards the document's end.
    
    Parameters:
    ----------
    pdf_directory : str
        The path to the directory containing PDF files to be processed. Each PDF file in this directory
        will be read, and its text will be cleaned and prepared for further processing.

    Returns:
    -------
    List[Tuple[str, Dict]]
        A list of tuples, where each tuple contains the cleaned text of a PDF file and a dictionary
        of metadata for that file. The metadata includes the file name, publication date, and any referenced
        websites extracted from the text.
    """
    cleaned_texts = []
    pdf_files = Path(pdf_directory).glob('*.pdf')
    
    for pdf_file in pdf_files:
        file_metadata = {'file_name': pdf_file.name}
        pdf_text = extract_and_process_text(str(pdf_file))
        file_metadata['publication_date'], pdf_text = extract_and_remove_all_dates(pdf_text)
        file_metadata['referenced_websites'], pdf_text = extract_and_remove_all_websites(pdf_text)
        pdf_text = remove_dot_sequences(pdf_text)
        pdf_text = remove_copyright_notice(pdf_text)
        pdf_text = scrub_text(pdf_text)
        pdf_text = remove_text_after_phrases(pdf_text)
        
        cleaned_texts.append((pdf_text, file_metadata))

    return cleaned_texts