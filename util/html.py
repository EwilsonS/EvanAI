"""HTML utilities."""

import re


def wrap_urls(text, anchor_config):
    """Wrap plain URLs in anchor tags"""
    url_pattern = r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]))"""  # pylint: disable=C0301 # noqa: E501
    existing_anchor_pattern = r"<a.*?href.*?<\/a>"

    # Find URLs that are already wrapped in anchor tags
    existing_anchors = re.findall(existing_anchor_pattern, text, re.IGNORECASE)

    # For each URL that is already wrapped, replace it with a placeholder in the text
    for i, anchor in enumerate(existing_anchors):
        placeholder = f"{{{{ANCHOR_{i}}}}}"
        text = text.replace(anchor, placeholder)

    # Find any remaining plain URLs that are not wrapped in anchor tags
    existing_plain_urls = [
        tup[0] for tup in re.findall(url_pattern, text, re.IGNORECASE)
    ]

    # For each plain URL, replace it with a placeholder in the text
    for i, plain_url in enumerate(existing_plain_urls):
        placeholder = f"{{{{PLAIN_URL_{i}}}}}"
        text = text.replace(plain_url, placeholder)

    # wrap keywords in anchor tags
    text = keyword_anchors(text, anchor_config)

    # Find keyword anchor tags
    existing_keyword_anchors = re.findall(existing_anchor_pattern, text, re.IGNORECASE)

    # For each keyword anchor, replace it with a placeholder in the text
    for i, keyword_anchor in enumerate(existing_keyword_anchors):
        placeholder = f"{{{{KEYWORD_ANCHOR_{i}}}}}"
        text = text.replace(keyword_anchor, placeholder)

    # Replace PLAIN_URL placeholders with the original plain URLs
    for i, url in enumerate(existing_plain_urls):
        placeholder = f"{{{{PLAIN_URL_{i}}}}}"
        text = text.replace(placeholder, url)

    # Wrap URLs with anchor tags in the text
    text = re.sub(
        url_pattern,
        r'<a href="\1" title="\1" rel="noopener" target="_blank">\1</a>',
        text,
    )

    # Replace placeholders with the original URLs that were already wrapped
    for i, anchor in enumerate(existing_anchors):
        placeholder = f"{{{{ANCHOR_{i}}}}}"
        text = text.replace(placeholder, anchor)

    # Replace placeholders with the original URLs that were already wrapped
    for i, anchor in enumerate(existing_keyword_anchors):
        placeholder = f"{{{{KEYWORD_ANCHOR_{i}}}}}"
        text = text.replace(placeholder, anchor)

    return text


def keyword_anchors(text, anchor_configs):
    """Create anchor tags from regex matches"""
    for config in anchor_configs:
        regex = (
            re.compile(config["regex"], re.IGNORECASE)
            if config.get("ignore_case", False)
            else re.compile(config["regex"])
        )
        replacement = f'<a href="{config["href"]}" title="{config["title"]}" rel="noopener" target="_blank">\\g<0></a>'
        text = regex.sub(replacement, text, count=config.get("count", 1))
    return text
