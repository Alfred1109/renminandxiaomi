#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Defines the structure for various report elements used in data-driven report generation.
"""

import pandas as pd
import os

class ReportElement:
    """Base class for all report elements."""
    def __init__(self, element_type: str):
        self.element_type = element_type

class TitleElement(ReportElement):
    """Represents a main title of the report."""
    def __init__(self, text: str):
        super().__init__('title')
        self.text = text

class HeadingElement(ReportElement):
    """Represents a section heading."""
    def __init__(self, level: int, text: str, element_id: str = None):
        super().__init__('heading')
        if not 1 <= level <= 6:
            raise ValueError("Heading level must be between 1 and 6.")
        self.level = level
        self.text = text
        self.element_id = element_id  # Useful for HTML anchors

class ParagraphElement(ReportElement):
    """Represents a paragraph of text."""
    def __init__(self, text: str):
        super().__init__('paragraph')
        self.text = text

class ImageElement(ReportElement):
    """Represents an image."""
    def __init__(self, path: str, caption: str = None, alt_text: str = None):
        super().__init__('image')
        self.path = path
        self.caption = caption
        self.alt_text = alt_text if alt_text else (os.path.basename(path) if path else "Image")

class TableElement(ReportElement):
    """Represents a table, preferably from a DataFrame."""
    def __init__(self, title: str = None, dataframe: pd.DataFrame = None, markdown_string: str = None):
        super().__init__('table')
        self.title = title
        self.dataframe = dataframe  # Primary source
        self.markdown_string = markdown_string  # Fallback if DataFrame is not available or for simple tables

class UnorderedListElement(ReportElement):
    """Represents an unordered list (bullet points)."""
    def __init__(self, items: list):
        super().__init__('unordered_list')
        # Items can be simple strings or other ReportElement objects for nested structures
        self.items = items

class OrderedListElement(ReportElement):
    """Represents an ordered list (numbered)."""
    def __init__(self, items: list):
        super().__init__('ordered_list')
        self.items = items

class RawTextElement(ReportElement):
    """For inserting pre-formatted text, e.g., Markdown snippets or HTML blocks."""
    def __init__(self, text: str):
        super().__init__('raw_text')
        self.text = text

class LineBreakElement(ReportElement):
    """Represents a line break or a horizontal rule depending on renderer."""
    def __init__(self, count: int = 1):
        super().__init__('linebreak')
        self.count = count # Number of line breaks, or could signify a horizontal rule if count is special

class CodeBlockElement(ReportElement):
    """Represents a block of code."""
    def __init__(self, code: str, language: str = None):
        super().__init__('code_block')
        self.code = code
        self.language = language

# Placeholder for a Table of Contents element, to be handled by renderers
class TOCElement(ReportElement):
    """Represents a placeholder for the Table of Contents."""
    def __init__(self):
        super().__init__('toc')

if __name__ == '__main__':
    # Example usage (for testing purposes)
    title = TitleElement("My Awesome Report")
    heading1 = HeadingElement(level=1, text="Introduction", element_id="intro")
    paragraph1 = ParagraphElement("This is the first paragraph of my awesome report.")
    
    # Simple list
    list_items = [
        "Item 1",
        "Item 2",
        ParagraphElement("Item 3 with a nested paragraph.") # Example of nested element
    ]
    u_list = UnorderedListElement(list_items)
    
    # Image
    img = ImageElement(path="output/figures/some_figure.png", caption="A very important figure.")
    
    # Table (example with DataFrame)
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    table = TableElement(title="Important Data", dataframe=df)

    report_doc = [title, heading1, paragraph1, u_list, img, table]

    for elem in report_doc:
        print(f"Element Type: {elem.element_type}, Content: {vars(elem)}") 