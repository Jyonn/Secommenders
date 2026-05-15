from process.base_amazon_processor import AmazonProcessor


class CDsProcessor(AmazonProcessor):
    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000

    def __init__(self, **kwargs):
        super().__init__(subset='CDs_and_Vinyl', **kwargs)
