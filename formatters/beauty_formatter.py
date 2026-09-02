from formatters.base_amazon_formatter import AmazonFormatter


class BeautyFormatter(AmazonFormatter):
    REQUIRE_STRINGIFY = True

    def __init__(self, **kwargs):
        super().__init__(subset='Beauty', **kwargs)
