import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
import re


# Function to fetch HTML content
def fetch_url_html_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        message = 'An unexpected error occurred while trying to fetch: ' + url
        print("\033[91m" + message + "\033[0m")
        return


# Function to parse HTML and extract text and links
def parse_html(html_content, main_url):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract text from all visible text-containing tags
    def get_visible_text(element):
        visible_text_elements = [
            'p', 'div', 'span', 'td', 'th', 'li', 'dd', 'dt',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ]
        if element.name in visible_text_elements and element.get_text(strip=True):
            return [element.get_text(strip=True)]
        return []

    texts = []
    body = soup.find('body')
    if body:
        for element in body.descendants:
            texts.extend(get_visible_text(element))

    # Extract links from the entire body
    def get_links_from_body(body):
        non_text_tags = ['script', 'style', 'meta', 'link', 'header', 'footer', 'nav', 'aside']
        links = []
        for descendant in body.descendants:
            if descendant.name not in non_text_tags:
                if descendant.name == 'a' and 'href' in descendant.attrs:
                    blink = descendant.attrs['href']
                    full_url = urljoin(main_url, blink)
                    full_url = normalize_url(full_url)
                    if (is_valid_url(full_url) and is_same_domain(main_url, full_url) and not is_pdf_url(full_url)
                            and not is_restricted_extension(full_url)):
                        links.append(full_url)
        return links

    data = []
    body = soup.find('body')
    if body:
        links = get_links_from_body(body)
        combined_text = ' '.join(texts).strip()
        if combined_text or links:
            data.append((combined_text, links))

    return data


# URL validation and normalization functions
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)


def is_same_domain(url1, url2):
    parsed_url1 = urlparse(url1)
    parsed_url2 = urlparse(url2)
    return parsed_url1.netloc == parsed_url2.netloc


def is_pdf_url(url):
    return url.lower().endswith('.pdf')


def is_restricted_extension(url):
    unwanted_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
                           '.mp4', '.avi', '.mov', '.wmv', '.flv', '.zip']
    return any(url.lower().endswith(ext) for ext in unwanted_extensions)


def normalize_url(url):
    parsed_url = urlparse(url)
    normalized_url = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
    if normalized_url.endswith('/') and parsed_url.path != '/':
        normalized_url = normalized_url[:-1]
    return normalized_url


# Recursive deep search function to examine desired domain websites
def deep_search_newlinks(parsed_data, MaxLinks):
    for text, links in parsed_data:

        for link in links:

            if link and (MaxLinks - len(CacheMapUrlsAndDatas)) > 0 and link not in CacheMapUrlsAndDatas:
                CacheMapUrlsAndDatas[link] = None
                try:
                    html_content = fetch_url_html_content(link)

                finally:  # After fetching the web page successfully, continue with this page
                    if html_content is not None:
                        subparsed_data = parse_html(html_content, link)
                        # Update the value of the link with parsed html to use later on.
                        CacheMapUrlsAndDatas[link] = subparsed_data

                        # Show the process
                        print("# of Remained Link:", MaxLinks - len(CacheMapUrlsAndDatas), ", and Fetched Link: ", link)

                        deep_search_newlinks(subparsed_data, MaxLinks)


# Function to check for specific keywords in text
def contains_euro_or_dollar(text):
    pattern = re.compile(r'\b(?:euro|dollar)s?\b|€|\$', re.IGNORECASE)
    return bool(pattern.search(text))


def contains_power_units(text):
    pattern = re.compile(
        r'\b(?:watt|kilowatt|kilowatts|megawatt|megawatts|gigawatt|gigawatts|kw|KW|mw|MW|gw|GW|kW|mW|gW)\b',
        re.IGNORECASE)
    return bool(pattern.search(text))

# Spacy component to extract solar park information
@spacy.Language.component("ExtractSolarParkInformationMethod")
def ExtractSolarParkInformationMethod(doc):
    investing_in_solarparks = False
    equity_check_sizes = []
    power_capacities = []

    # Define patterns for identifying equity check sizes and power capacities
    matcher = Matcher(doc.vocab)
    # Patterns for EQUITY_CHECK_SIZE values
    matcher.add("EQUITY_CHECK_SIZE", [
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["euros", "dollars"]}}],
        [{"LOWER": {"IN": ["€", "$"]}}, {"LIKE_NUM": True}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion"]}}, {"LOWER": {"IN": ["euros", "dollars"]}}],
        [{"LOWER": {"IN": ["€", "$"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion"]}}],
        [{"LOWER": {"IN": ["over", "above"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion"]}},
         {"LOWER": {"IN": ["euros", "dollars"]}}],
    ])
    # Patterns for power capacities
    matcher.add("POWER_CAPACITY", [
        [{"LIKE_NUM": True}, {"LOWER": {
            "IN": ["kw", "mw", "gw", "kilowatt", "megawatt", "gigawatt", "kilowatts", "megawatts", "gigawatts"]}}],
        [{"LOWER": {"IN": ["over", "above"]}}, {"LIKE_NUM": True},
         {"LOWER": {"IN": ["kw", "mw", "gw", "kilowatt", "megawatt", "gigawatt"]}}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion"]}},
         {"LOWER": {"IN": ["kw", "mw", "gw", "kilowatt", "megawatt", "gigawatt"]}}],
        [{"LOWER": {"IN": ["over", "above"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion"]}},
         {"LOWER": {"IN": ["kw", "mw", "gw", "kilowatt", "megawatt", "gigawatt"]}}]
    ])

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        if doc.vocab.strings[match_id] == "EQUITY_CHECK_SIZE":
            equity_check_sizes.append(span.text)
        elif doc.vocab.strings[match_id] == "POWER_CAPACITY":
            power_capacities.append(span.text)

    # Check if there are keywords indicating investment in solar projects
    solar_keywords = ['solarpark', 'solar', 'renewable', 'energy', 'wind', 'solarparks']
    keywords_found = [token.text.lower() for token in doc if token.text.lower() in solar_keywords]

    if any(keyword in keywords_found for keyword in ['solarpark', 'solar', 'solarparks']):
        investing_in_solarparks = True

    # Remove duplicated items and check item again by using regex
    equity_check_sizes = [item for item in equity_check_sizes if contains_euro_or_dollar(item)]
    power_capacities = [item for item in power_capacities if contains_power_units(item)]
    equity_check_sizes = list(set(equity_check_sizes))
    power_capacities = list(set(power_capacities))

    # Set the values to the document extensions
    doc._.investing_in_solarparks = investing_in_solarparks
    doc._.equity_check_sizes = equity_check_sizes
    doc._.power_capacities = power_capacities

    return doc


# Function to analyze text using Spacy by defining pattern
def AnalyzeTextNLPPatternM(text):
    # Process the text
    doc = nlp(text)

    # Access the extracted information
    investing_in_solarparks = doc._.investing_in_solarparks
    equity_check_sizes = doc._.equity_check_sizes
    power_capacities = doc._.power_capacities
    #Output
    if investing_in_solarparks:
        return "Yes", equity_check_sizes, power_capacities
    else:
        return "No", "N/A", "N/A"


# Function to analyze text using Spacy by using predefined model
def AnalyzeTextNLPPredefinedM(text):
    doc = nlp(text)

    investment_amounts = []
    capacities = []
    is_solarpark_investment = False

    for token in doc:
        if "solarpark" in token.text.lower() or "solar" in token.text.lower() or "solar energy" in token.text.lower():
            is_solarpark_investment = True
            break

    if is_solarpark_investment:
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                investment_amounts.append(ent.text)

        # If no monetary amounts are found, attempt to extract them from text directly
        if not investment_amounts:
            for i, token in enumerate(doc):
                if token.like_num:  # Check if the token is a numerical value
                    if i + 1 < len(doc):
                        next_token = doc[i + 1]
                        if next_token.text.lower() in ["million", "billion", "thousand", "hundred"]:
                            if i + 2 < len(doc):
                                after_next_token = doc[i + 2]
                                if after_next_token.text.lower() in ["dollars", "euros"]:
                                    investment_amounts.append(f"{token.text} {next_token.text} {after_next_token.text}")
                                    break  # Only take the first occurrence

                        if next_token.text.lower() in ["dollars", "euros"]:
                            investment_amounts.append(f"{token.text} {next_token.text}")
                            break

        # Iterate over tokens to find capacity values
        for i, token in enumerate(doc):
            if token.like_num:  # Check if the token is a numerical value
                if i + 1 < len(doc):
                    next_token = doc[i + 1]
                    if next_token.text in ["MW", "megawatts", "KW", "kilowatts", "GW", "gigawatts", "kW", "mW"]:
                        if i > 0 and token.text.lower() in ["million", "billion", "thousand", "hundred"]:
                            capacities.append(f"{doc[i - 1].text} {token.text} {next_token.text}")
                        else:
                            capacities.append(f"{token.text} {next_token.text}")
                    elif next_token.text.lower() in ["megawatt", "kilowatt", "gigawatt", "watt"]:
                        if i > 0 and token.text.lower() in ["million", "billion", "thousand", "hundred"]:
                            capacities.append(f"{doc[i - 1].text} {token.text} {next_token.text}")
                        else:
                            capacities.append(f"{doc[i - 1].text} {token.text} {next_token.text.lower()}s")

    # Remove duplicated items and check item again by using regex
    investment_amounts = [item for item in investment_amounts if contains_euro_or_dollar(item)]
    capacities = [item for item in capacities if contains_power_units(item)]

    investment_amounts = list(set(investment_amounts))
    capacities = list(set(capacities))

    if is_solarpark_investment:
        return "Yes", investment_amounts, capacities
    else:
        return "No", "N/A", "N/A"

# This function displays the results
def display_results_as_tables(results):
    for result in results:
        df = pd.DataFrame([{
            'Equity Check Size': result['Equity Check Size'],
            'Power Capacities': result['Capacities'],
            'Investing in Solarparks': result['Investing in Solarparks']
        }])
        print(f"\nCompany: {result['Company']}")
        print(df.to_string(index=False))
        print("\nWebsite:", result['Website'])
        print("\n-----------------------")


# Task function
def RunTask(MaxLinks=1, NLPMethod="0"):
    company_urls = {
        'ENREGO Energy GmbH': 'https://enrego.de/en',
        'ENVIRIA Energy Holding GmbH': 'https://enviria.energy/en/cases/maja-furniture-factory-energy-as-a-service',
        'HIH Invest Real Estate Austria GmbH': 'https://hih-invest.de/en/',
        'Merkle Germany GmbH': 'https://www.merkle.com/dach/en'
    }
    results = []

    def is_duplicate_entry(results, company, infochecksize, infocapacity, investing_in_solarparks):
        for entry in results:
            if (entry['Company'] == company and
                    entry['Equity Check Size'] == infochecksize and
                    entry['Capacities'] == infocapacity and
                    entry['Investing in Solarparks'] == investing_in_solarparks):
                return True
        return False

    for company, url in company_urls.items():
        html_content = fetch_url_html_content(url)

        companyAdded = False
        if html_content:
            parsed_data = parse_html(html_content, url)
            if url not in CacheMapUrlsAndDatas:
                CacheMapUrlsAndDatas[url] = parsed_data

            print("# of Remained Link:", MaxLinks - len(CacheMapUrlsAndDatas), ", and Fetched Link: ", url)
            # This process may take time
            deep_search_newlinks(parsed_data, MaxLinks)

            print(f"Processing data for {company}")
            for link, each_data in CacheMapUrlsAndDatas.items():
                if each_data and isinstance(each_data, list):
                    for text, links in each_data:
                        if text:
                            try:
                                # Use NLP Match to analyze the text.
                                result = AnalyzeTextNLPPredefinedM(text)

                                if result[0] == "Yes":
                                    companyAdded = True
                                    infochecksize = result[1]
                                    infocapacity = result[2]

                                    if not is_duplicate_entry(results, company, infochecksize, infocapacity, result[0]):
                                        results.append({
                                            'Company': company,
                                            'Equity Check Size': infochecksize,
                                            'Capacities': infocapacity,
                                            'Investing in Solarparks': result[0],
                                            'Website': link
                                        })

                            except Exception as e:
                                print(f"Error processing text for {link}: {e}")
                else:
                    print(f"each_data is not a list or is empty for link: {link}")

            if not companyAdded:
                results.append({
                    'Company': company,
                    'Equity Check Size': 'N/A',
                    'Capacities': 'N/A',
                    'Investing in Solarparks': 'N/A',
                    'Website': url
                })
            CacheMapUrlsAndDatas.clear()
        else:
            results.append({
                'Company': company,
                'Equity Check Size': 'N/A',
                'Capacities': 'N/A',
                'Investing in Solarparks': 'No',
                'Website': url
            })

    display_results_as_tables(results)


# Test function
def RunTest(nlp_method):
    examples = [
        (
            "WindGen Corporation, a leader in renewable energy, has announced a substantial investment in wind energy projects. "
            "They are allocating above €10 million to develop wind farms in coastal regions known for high wind speeds. "
            "These farms will have turbines with capacities ranging from 2 MW to 10 MW each. WindGen aims to harness the power "
            "of wind to generate clean energy for thousands of homes and businesses. The first phase of the project includes "
            "installing 20 turbines by the end of next year, with plans for further expansion in the coming years."
        ),
        (
            "SolarPower Inc. has invested 50 million dollars in solarparks projects. The new solarparks are expected to generate "
            "over 100 MW of power, providing electricity to thousands of households. This investment marks a significant step "
            "towards sustainable energy solutions."
        ),
        (
            "Green Energy Solutions is planning to invest $20 million in renewable energy. Their focus is on building solarparks "
            "and wind farms with a combined capacity of 500 megawatts (MW). This initiative aims to reduce carbon emissions and "
            "promote clean energy."
        ),
        (
            "Renewable Future Corp is investing 30 million euros in solar and wind energy projects. The new plants will have a "
            "capacity of 200 MW each, contributing significantly to the company's renewable energy portfolio."
        ),
        (
            "Sunshine Renewables Ltd. has committed to funding a large-scale solarpark initiative in the desert region of Arizona. "
            "They plan to invest over $100 million in setting up solarparks with a total capacity of 1 GW (gigawatt). This ambitious "
            "project aims to provide clean energy to cities across the southwestern United States."
        ),
        (
            "EcoPower Solutions, a startup specializing in renewable energy, is seeking investors for their solarpark expansion "
            "project. They aim to raise $50 million to increase their solarpark capacity by 300 MW. This expansion will enhance "
            "their ability to meet growing demand for clean energy in urban areas."
        ),
        (
            "Global Energy Ventures, a multinational corporation, has announced a partnership with local governments to develop "
            "solarpark projects in multiple countries. They have earmarked $150 million for these projects, which are expected to "
            "generate a combined capacity of 500 MW. This initiative is part of their commitment to sustainable development goals."
        ),
        (
            "Wind and Solar Innovations Ltd. has secured funding of €80 million to expand their renewable energy portfolio. "
            "They are focusing on both wind farms and solarparks, with plans to develop projects with capacities ranging from "
            "10 MW to 50 MW each. This investment underscores their commitment to clean energy solutions."
        )
    ]
    # Show the results of examples for method Pattern
    for example in examples:
        if nlp_method == "0":
            result = AnalyzeTextNLPPatternM(example)
        else:
            result = AnalyzeTextNLPPredefinedM(example)
        # Output the results
        print("Text Example:", example)
        print("Investing in Solarparks:", result[0])
        print("Equity Check Sizes:", result[1])
        print("Power Capacities:", result[2])
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":

    CacheMapUrlsAndDatas = {}

    user_choice_method = input("Do you want to run method Pattern[0] or method Predefined[1] for NLP? ")

    if user_choice_method == '0':
        # Define the custom extension attributes
        Doc.set_extension('investing_in_solarparks', default=False, force=True)
        Doc.set_extension('equity_check_sizes', default=[], force=True)
        Doc.set_extension('power_capacities', default=[], force=True)

    # Load the language model
    nlp = spacy.load('en_core_web_sm')

    if user_choice_method == '0':
        # Add a component to the pipeline
        if 'ExtractSolarParkInformationMethod' not in nlp.pipe_names:
            nlp.add_pipe("ExtractSolarParkInformationMethod", last=True)

    user_choice_tt = input("Do you want to run Task[0] or Test[1]? ")

    if user_choice_tt == '0':
        max_limit = int(input("Please indicate number of websites search per company(Max. allowed limit is 50):  "))
        max_limit = 50 if max_limit > 50 else max_limit
        RunTask(max_limit, user_choice_method)
    else:
        RunTest(user_choice_method)
