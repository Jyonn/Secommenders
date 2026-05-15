from processors.base_amazon_processor import AmazonProcessor


class CDsProcessor(AmazonProcessor):
    def __init__(self, **kwargs):
        super().__init__(subset='CDs_and_Vinyl', **kwargs)
