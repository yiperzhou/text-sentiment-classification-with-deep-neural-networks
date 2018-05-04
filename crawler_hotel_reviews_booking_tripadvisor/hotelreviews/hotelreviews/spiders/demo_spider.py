import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        'https://www.booking.com/reviews/fi/hotel/haven.html'
    ]

    def parse(self, response):
        for rev_box in response.xpath('//li[@class="review_item clearfix"]'):
            yield {
                'score': rev_box.xpath('//div[@class="review_item_review_score jq_tooltip"]/text()').extract_first(),
                'author': quote.css('small.author::text').extract_first(),
                'tags': quote.css('div.tags a.tag::text').extract(),
            }