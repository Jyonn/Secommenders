from formatters.base_amazon_formatter import AmazonFormatter


class BeautyFormatter(AmazonFormatter):
    REQUIRE_STRINGIFY = True
    SUBSET_ALIASES = ('Beauty',)

    def __init__(self, **kwargs):
        super().__init__(subset='All_Beauty', **kwargs)
