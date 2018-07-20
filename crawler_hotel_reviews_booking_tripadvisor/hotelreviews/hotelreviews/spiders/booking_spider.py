from __future__ import absolute_import
import scrapy
from scrapy.loader import ItemLoader
from hotelreviews.items import BookingReviewItem

#crawl up to 6 pages of review per hotel
# max_pages_per_hotel = 100

class BookingSpider(scrapy.Spider):
    name = "booking"
    start_urls = [
        #"https://www.booking.com/searchresults.html?city=-1364995" #helsinki
        # "https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNiBW5vcmVmaEiIAQGYATHCAQN4MTHIAQzYAQPoAQH4AQKSAgF5qAID;sid=9e01ded3f25b53e4ff5999b711ed08fe;checkin_month=7&checkin_monthday=5&checkin_year=2017&checkout_month=7&checkout_monthday=6&checkout_year=2017&class_interval=1&dest_id=-1456928&dest_type=city&dtdisc=0&group_adults=2&group_children=0&inac=0&index_postcard=0&label_click=undef&no_rooms=1&offset=0&postcard=0&raw_dest_type=city&room1=A%2CA&sb_price_type=total&search_selected=1&src=index&src_elem=sb&ss=Paris%2C%20Ile%20de%20France%2C%20France&ss_all=0&ss_raw=Paris&ssb=empty&sshis=0&"#Paris
        #London
       # "https://www.booking.com/searchresults.html?label=gen173nr-1FCAEoggJCAlhYSDNiBW5vcmVmaEiIAQGYATHCAQN4MTHIAQzYAQHoAQH4AQKSAgF5qAID;sid=9e01ded3f25b53e4ff5999b711ed08fe;class_interval=1&dest_id=-1746443&dest_type=city&dtdisc=0&group_adults=2&group_children=0&inac=0&index_postcard=0&label_click=undef&no_rooms=1&offset=0&postcard=0&raw_dest_type=city&room1=A%2CA&sb_price_type=total&search_selected=1&src=index&src_elem=sb&ss=Berlin%2C%20Berlin%20Federal%20State%2C%20Germany&ss_all=0&ss_raw=Be&ssb=empty&sshis=0&ssne=Paris&ssne_untouched=Paris&" #Berlin
        #
        "https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggJCAlhYSDNiBW5vcmVmaEiIAQGYATHCAQN4MTHIAQzYAQPoAQH4AQKSAgF5qAID;sid=5bbd1c1ae5a8c3b6a7dcce7d8431cdcc;class_interval=1&dest_id=-3212216&dest_type=city&dtdisc=0&group_adults=2&group_children=0&inac=0&index_postcard=0&label_click=undef&no_rooms=1&offset=0&postcard=0&raw_dest_type=city&room1=A%2CA&sb_price_type=total&search_selected=1&src=index&src_elem=sb&ss=R%C4%ABga%2C%20Vidzeme%2C%20Latvia&ss_all=0&ss_raw=Riga&ssb=empty&sshis=0&"
    ]

    # pageNumber = 1

    def parse(self, response):
        # if self.hotelcount > 3000:
        #     return
        for hotelurl in response.xpath('//a[@class="hotel_name_link url"]/@href'):
            url = response.urljoin(hotelurl.extract())
            yield scrapy.Request(url, callback=self.parse_hotel)

        next_page = response.xpath('//a[starts-with(@class,"paging-next")]/@href')
        if next_page:
            url = response.urljoin(next_page[0].extract())
            yield scrapy.Request(url, self.parse)

    #get its reviews page
    def parse_hotel(self, response):
        reviewsurl = response.xpath('//a[@class="show_all_reviews_btn"]/@href')
        url = response.urljoin(reviewsurl.extract_first())

        yield scrapy.Request(url, callback=self.parse_reviews)


    #and parse the reviews
    def parse_reviews(self, response):

        for rev in response.xpath('//div[@class="review_item_header_content_container"]/a/@href').extract():
            url = response.urljoin(rev)
            yield scrapy.Request(url, callback= self.parse_single_review)

        next_page = response.xpath('//a[@id="review_next_page_link"]/@href')
        if next_page:
            url = response.urljoin(next_page.extract_first())
            yield scrapy.Request(url, self.parse_reviews)

    def parse_single_review(self, response):

        item = BookingReviewItem()
        item['title'] = response.xpath('//div[@class="review_item_header_content_container"]/div/span/text()').extract_first()
        item['score'] = response.xpath('//div[@class="review_item_review_score jq_tooltip"]/text()').extract_first()
        item['negative_content'] = response.xpath('.//p[@class="review_neg"]//span/text()').extract_first()
        item['positive_content'] = response.xpath('.//p[@class="review_pos"]//span/text()').extract_first()
        item['date'] = response.xpath('.//p[@class="review_item_date"]/text()').extract_first()
        item["hotelName"] = response.xpath('//h1[@class="reviews_review_hotel_name"]/text()').extract_first()
        item["hotelStars"] = response.xpath('//span[@class="invisible_spoken"]/text()').extract_first()
        item["url"] = response.url
        item["hotelUrl"] = response.urljoin(response.xpath('//button[@class="b-button b-button_primary"]/@href').extract_first())
        item["hotelLocation"] = response.xpath('//p[@class="reviews_review_hotel_address"]/text()').extract_first()

        yield item
