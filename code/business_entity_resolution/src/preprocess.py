"""
Preprocessing module for Business Entity Resolution.

Normalizes business names, addresses, and country values to reduce noise
from abbreviations, punctuation, transliteration variants, and formatting
differences.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional

# Common legal entity suffixes to normalize
SUFFIX_MAP = {
    'llc': 'limited liability company',
    'l.l.c': 'limited liability company',
    'llp': 'limited liability partnership',
    'l.l.p': 'limited liability partnership',
    'plc': 'public limited company',
    'pvt': 'private',
    'pvt.': 'private',
    'priv': 'private',
    'private': 'private',
    'limited': 'limited',
    'ltd': 'limited',
    'ltd.': 'limited',
    'inc': 'incorporated',
    'inc.': 'incorporated',
    'incorporated': 'incorporated',
    'corp': 'corporation',
    'corp.': 'corporation',
    'corporation': 'corporation',
    'co': 'company',
    'co.': 'company',
    'company': 'company',
    'llc': 'limited liability company',
    'lp': 'limited partnership',
    'l.p': 'limited partnership',
    'llc': 'limited liability company',
    'pa': 'professional association',
    'p.a': 'professional association',
    'pc': 'professional corporation',
    'p.c': 'professional corporation',
    'llp': 'limited liability partnership',
}

# Common business stopwords to optionally remove
BUSINESS_STOPWORDS = {
    'the', 'and', '&', 'of', 'for', 'in', 'on', 'at', 'to', 'from', 'with',
    'by', 'de', 'la', 'le', 'les', 'del', 'von', 'van', 'der', 'di', 'da',
}

# Address abbreviation mappings
ADDRESS_ABBR_MAP = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'ln': 'lane', 'dr': 'drive', 'ct': 'court', 'cir': 'circle',
    'pl': 'place', 'plz': 'place', 'ste': 'suite', 'apt': 'apartment',
    'unit': 'unit', 'hwy': 'highway', 'pkwy': 'parkway', 'trl': 'trail',
    'cv': 'cove', 'ct': 'court', 'de': 'delaware', 'st': 'street',
    'mt': 'mount', 'ft': 'fort', 'n': 'north', 's': 'south',
    'e': 'east', 'w': 'west', 'ne': 'northeast', 'nw': 'northwest',
    'se': 'southeast', 'sw': 'southwest', 'po': 'post office',
    'rte': 'route', 'rm': 'room', 'fl': 'floor', 'frnt': 'front',
    'hwy': 'highway', 'intl': 'international', 'est': 'estate',
    'pk': 'park', 'dept': 'department', 'ctr': 'center', 'cnt': 'center',
    'hlth': 'health', 'dept': 'department', 'div': 'division',
    'bch': 'beach', 'cde': 'code', 'cst': 'coast', 'dl': 'delaware',
    'ga': 'georgia', 'hi': 'hawaii', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota',
    'ms': 'mississippi', 'mo': 'missouri', 'mt': 'montana',
    'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire',
    'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio',
    'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota',
    'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'vt': 'vermont',
    'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming',
}

# Country name normalizations
COUNTRY_MAP = {
    'us': 'US',
    'usa': 'US',
    'united states': 'US',
    'united states of america': 'US',
    'india': 'India',
    'in': 'India',
    'france': 'France',
    'fr': 'France',
    'republic of india': 'India',
}


def normalize_text(text: str) -> str:
    """
    Basic text normalization: unicode normalization, lowercase, 
    punctuation removal, whitespace collapsing.
    """
    if not text or (isinstance(text, float) and pd.isna(text)):
        return ''

    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    # Remove non-ASCII characters but keep spaces
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()
    # Replace punctuation with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name(name: str) -> str:
    """
    Normalize a business name for matching.
    
    - Unicode normalization
    - Lowercase
    - Remove punctuation
    - Expand common suffixes
    - Remove extra whitespace
    - Remove common stopwords
    """
    text = str(name).strip() if name else ''
    if not text:
        return ''

    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()

    # Replace special characters with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normalize common suffixes
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        if token in SUFFIX_MAP:
            normalized_tokens.append(SUFFIX_MAP[token])
        else:
            normalized_tokens.append(token)

    text = ' '.join(normalized_tokens)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name_compact(name: str) -> str:
    """
    Compact normalization (no stopword removal) for token-based blocking.
    Preserves all significant tokens.
    """
    text = str(name).strip() if name else ''
    if not text:
        return ''

    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name_compact_series(series: pd.Series) -> pd.Series:
    """Vectorized version of normalize_name_compact for pandas Series."""
    text = series.astype(str).str.strip()
    text = text.apply(lambda x: unicodedata.normalize('NFKD', x))
    text = text.str.encode('ascii', 'ignore').str.decode('ascii')
    text = text.str.lower().str.strip()
    text = text.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    return text


def normalize_address(addr: str) -> str:
    """
    Normalize a business address for matching.
    
    - Unicode normalization
    - Lowercase
    - Expand common abbreviations
    - Remove extra whitespace
    - Handle null/missing values
    """
    text = str(addr).strip() if addr else ''
    if not text or text.lower() in ('nan', 'null', 'none', ''):
        return ''

    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()

    # Remove bracketed content like [INC] (INC) etc.
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\([^)]*\)', ' ', text)

    # Replace special characters with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normalize abbreviations
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        if token in ADDRESS_ABBR_MAP:
            normalized_tokens.append(ADDRESS_ABBR_MAP[token])
        else:
            normalized_tokens.append(token)

    text = ' '.join(normalized_tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_tokens(text: str) -> Set[str]:
    """Extract significant tokens from text."""
    if not text:
        return set()
    tokens = set()
    for token in text.split():
        if len(token) > 1 and token not in BUSINESS_STOPWORDS:
            tokens.add(token)
    return tokens


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text (with word boundaries)."""
    if not text or len(text) < n:
        return set()
    text = text.lower().strip()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def normalize_country(country: str) -> str:
    """
    Normalize country value. Handles open-set countries.
    Does NOT hard-code US/India - any country string is preserved.
    """
    if not country or (isinstance(country, float)):
        return ''

    text = str(country).strip()
    if not text or text.lower() in ('nan', 'null', 'none', ''):
        return ''

    lower = text.lower()
    if lower in COUNTRY_MAP:
        return COUNTRY_MAP[lower]
    # Return the original value (open set) - don't restrict to known countries
    return text


def normalize_country_series(series: pd.Series) -> pd.Series:
    """Vectorized version of normalize_country for pandas Series."""
    lower = series.astype(str).str.lower().str.strip()
    is_null = lower.isin(['nan', 'null', 'none', ''])
    mapped = lower.map(COUNTRY_MAP)
    result = mapped.fillna(series.astype(str).str.strip())
    result = result.where(~is_null, '')
    return result


def extract_postal_code(address: str) -> str:
    """Extract postal/ZIP code from address string."""
    if not address:
        return ''
    text = str(address).strip()
    # US ZIP codes
    zip_match = re.search(r'\b(\d{5})\b', text)
    if zip_match:
        return zip_match.group(1)
    # Indian PIN codes
    pin_match = re.search(r'\b(\d{6})\b', text)
    if pin_match:
        return pin_match.group(1)
    # French postal codes
    fpost_match = re.search(r'\b(\d{5})\b', text)
    if fpost_match:
        return fpost_match.group(1)
    return ''


def extract_house_number(address: str) -> str:
    """Extract house/building number from address string."""
    if not address:
        return ''
    text = str(address).strip()
    # Match patterns like "123", "H.No. 123", "Flat No. 123"
    match = re.search(r'(?:h\.?no\.?|house no\.?|flat no\.?|no\.?|#)[\s]*(\d+[a-zA-Z/\-]*)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Simple: first number in address
    match = re.match(r'\s*(\d+[a-zA-Z/\-]*)', text)
    if match:
        return match.group(1).strip()
    return ''


def normalize_pair_name(name: str) -> str:
    """Alias for normalize_name_compact - used for token blocking key."""
    return normalize_name_compact(name)


def preprocess_entity(name: str, address: str, country: str) -> dict:
    """Preprocess a single entity and return all normalized fields."""
    return {
        'normalized_name': normalize_name(name),
        'token_name': normalize_name_compact(name),
        'normalized_address': normalize_address(address),
        'country': normalize_country(country),
        'name_tokens': extract_tokens(normalize_name_compact(name)),
        'postal_code': extract_postal_code(address),
        'house_number': extract_house_number(address),
    }


import pandas as pd

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to a DataFrame of entities."""
    result = df.copy()
    result['normalized_name'] = df['business_name'].apply(normalize_name_compact)
    result['normalized_address'] = df['business_address'].apply(normalize_address)
    result['country_norm'] = df['country'].apply(normalize_country)
    result['name_tokens'] = result['normalized_name'].apply(lambda x: extract_tokens(x))
    result['postal_code'] = df['business_address'].apply(extract_postal_code)
    result['house_number'] = df['business_address'].apply(extract_house_number)
    return result
