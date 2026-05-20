from formatters.base_amazon_formatter import AmazonFormatter


class CDsFormatter(AmazonFormatter):
    def __init__(self, **kwargs):
        super().__init__(subset='CDs_and_Vinyl', **kwargs)
