import re


def stocktwit_cleanup(text, ticker):
    if text.startswith("$" + ticker):
        text = remove_links(text)
        text = replace_tickers(text)
        if len(re.sub("(\[TICKER\])|(\[LINK\])", "", text)) <= 5:
            return None
        else:
            return text
    else:
        return None


def twitter_cleanup(text):
    text = remove_links(text)
    text = replace_tickers(text)
    text = remove_ats(text)
    text = remove_hash(text)
    if len(re.sub("(\[TICKER\])|(\[LINK\])", "", text)) <= 5:
        return None
    else:
        return text


def remove_links(text):
    return re.sub("(http|https):\/\/(\w|\d|\.|\/|\-|\?|\=)+(?=\s)?", "[LINK]", text)


def replace_tickers(text):
    return re.sub("\$[A-Z]+", "[TICKER]", text)


def remove_ats(text):
    return re.sub("[@]\w+", "", text)


def remove_hash(text):
    return re.sub("[#]", "", text)
