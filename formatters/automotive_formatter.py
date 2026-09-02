from formatters.base_amazon_formatter import AmazonFormatter


class AutomotiveFormatter(AmazonFormatter):
    def __init__(self, **kwargs):
        super().__init__(subset='Automotive', **kwargs)
