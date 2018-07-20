from __future__ import absolute_import
import scrapy
from scrapy.loader import ItemLoader
# from hotelreviews.items import BookingReviewItem

class BookingReviewsSpider(scrapy.Spider):
    name = "bookingreviews"
    start_urls = [
        "https://www.booking.com/reviews/fi/hotel/haven.html"
    ]

    def parse(self, response):

        for rev_box in response.xpath('//li[@class="review_item clearfix"]'):
            item = dict()
            item['score'] = rev_box.xpath('//div[@class="review_item_review_score jq_tooltip"]/text()').extract_first()
            item['negative_content'] = rev_box.xpath('.//p[@class="review_neg"]//span/text()').extract_first()
            item['positive_content'] = response.xpath('.//p[@class="review_pos"]//span/text()').extract_first()
            item['date'] = response.xpath('.//p[@class="review_item_date"]/text()').extract_first()

            yield item

        next_page = response.xpath('//a[@id="review_next_page_link"]/@href')
        if next_page:
            url = response.urljoin(next_page[0].extract())
            yield scrapy.Request(url, self.parse)