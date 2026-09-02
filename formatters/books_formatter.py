from formatters.base_amazon_formatter import AmazonFormatter


class BooksFormatter(AmazonFormatter):
    def __init__(self, **kwargs):
        super().__init__(subset='Books', **kwargs)
