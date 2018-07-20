import scrapy
from hotelreviews.items import TripadvisorReviewItem
from selenium import webdriver
from bs4 import BeautifulSoup

#TODO use loaders
class TripadvisorSpider(scrapy.Spider):
    name = "tripadvisor"
    start_urls = [
        # "https://www.tripadvisor.com/Hotels-g188590-Amsterdam_North_Holland_Province-Hotels.html"
        # "https://www.tripadvisor.com/Hotels-g189852-Stockholm-Hotels.htmls"
        # "https://www.tripadvisor.com/Hotels-g189948-Tampere_Pirkanmaa-Hotels.html"
        # "https://www.tripadvisor.com/Hotels-g189948-Tampere_Pirkanmaa-Hotels.html"
        # "https://www.tripadvisor.com/Hotels-g189934-Helsinki_Uusimaa-Hotels.html"
        # "https://www.tripadvisor.com/Hotel_Review-g187147-d188738-Reviews-Le_Pavillon_de_la_Reine-Paris_Ile_de_France.html"
        "https://www.tripadvisor.com/Hotels-g189934-Helsinki_Uusimaa-Hotels.html"
    ]
    def __init__(self):
        self.firstPage = True

    def parse(self, response):
        for href in response.xpath('//div[@class="listing_title"]/a/@href'):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_hotel)

        next_page = response.xpath('//div[@class="unified pagination standard_pagination"]/child::*[2][self::a]/@href')
        if next_page:
            url = response.urljoin(next_page.extract_first())
            yield scrapy.Request(url, self.parse)
        # yield scrapy.Request(response.url, self.parse_hotel)

    def parse_hotel(self, response):

        # for href in response.xpath('//div[starts-with(@class,"quote")]/a/@href'):
        #     url = response.urljoin(href.extract())
        #     yield scrapy.Request(url, callback=self.parse_review)


        # next_page_div = response.xpath('//div[@class="unified pagination north_star "]').extract_first()
        # try:

        # next_page_button = self.next_reviews_page.find_element_by_xpath("//span[@class='tabs_header reviews_header block_title']")
        # next_page_button.click()
        # except:
        #     print "error next click"


        # next_page = next_page_div.xpath('//span[@class="nav next taLnk "][self::a]/@href')
        # next_page = response.xpath('//div[@class="unified pagination north_star "]/child::*[2][self::a]/@href')
        # # next_page = response.xpath("//")
        # if next_page:
        #     url = response.urljoin(next_page.extract_first())
        #     yield scrapy.Request(url, self.parse_hotel)

    #   next_page = response.xpath('//div[@class="unified pagination "]/child::*[2][self::a]/@href')
    #     next_page = response.xpath('//div[@class="unified pagination north_star "]/span[starts-with(@class, "nav next")]').extract_first()
    #     # construct a new URL to get next page reviews for this hotel, after analysizing url in chrome,
    #     # the original url is like: https://www.tripadvisor.com/Hotel_Review-g189934-d652575-Reviews-Hotel_Katajanokka-Helsinki_Uusimaa.html
    #     # the next page review url is like: https://www.tripadvisor.com/Hotel_Review-g189934-d652575-Reviews-or5-Hotel_Katajanokka-Helsinki_Uusimaa.html
    #
    #     if next_page:
    #         urlparts = response.url.split("-Reviews-")
    #         next_page_review_url = urlparts[0] + str("-Reviews-or") + str(self.reviewPage * 5) + str("-") + urlparts[1]
    #         # url = response.urljoin(next_page.extract_first())
    #         self.reviewPage += 1
    #         yield scrapy.Request(next_page_review_url, self.parse_hotel)
    #     else:
    #         print "current review Page %s", self.reviewPage
    #     if self.firstPage:
        total_page_num = response.xpath('//div[@class="pageNumbers"][1]/span[last()]/text()').extract_first()
        if total_page_num:
            for reviewPage in range(1, int(total_page_num), 1):
                urlparts = response.url.split("-Reviews-")
                next_page_review_url = urlparts[0] + str("-Reviews-or") + str(reviewPage * 5) + str("-") + urlparts[1]
                yield scrapy.Request(next_page_review_url, self.parse_review_page)
            # self.firstPage = False
        yield scrapy.Request(response.url, self.parse_review_page)


    def parse_review_page(self, response):
        for href in response.xpath('//div[starts-with(@class,"quote")]/a/@href'):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_review)





    #to get the full review content I open its page, because I don't get the full content on the main page
    #there's probably a better way to do it, requires investigation
    def parse_review(self, response):
        item = TripadvisorReviewItem()
        try:

            item['title'] = response.xpath('//div[@class="quote"]/text()').extract()[0][1:-1] #strip the quotes (first and last char)
        except:
            item['title'] = None
        try:
            reviewContent = response.xpath('//div[@class="entry"]/p[1]').extract_first()
            soup = BeautifulSoup(reviewContent, "lxml")
            item['review'] = soup.find("p").text.strip()
        except:
            item["review"] = None

        try:
            score = response.xpath('//div[@class="rating reviewItemInline"]/span/@class').extract_first()
            score = float(score.split("_")[-1])/10
            item['score'] = score
        except:
            score = None
            item['score'] = score

        item["url"] = response.url
        try:
            item["date"] = response.xpath('//span[@class="ratingDate relativeDate"]/@title').extract_first()
        except:
            item["date"] = None
        try:
            item["hotelName"] = response.xpath('//div[@class="surContent"]/a[@class="HEADING"]/text()').extract_first()
        except:
            item["hotelName"] = None
        try:
            item["hotelUrl"] = response.urljoin(response.xpath('//div[@class="surContent"]/a[@class="HEADING"]/@href').extract_first())
        except:
            item["hotelUrl"] = None
        try:
            street_address = response.xpath('//span[@class="street-address"]/text()').extract_first()
            locality = response.xpath('//span[@class="locality"]/text()').extract_first()
            country_name = response.xpath('//span[@class="country-name"]/text()').extract_first()

            item["hotelLocation"] = street_address + ", " + locality + country_name
        except:
            item["hotelLocation"] = None
        try:
            item["hotelStars"] = response.xpath('//span[@class="star"]/span/img/@alt').extract_first().split()[0]
        except:
            item["hotelStars"] = None
        try:
            item["userId"] = response.xpath('//div[@class="username mo"]/span/text()').extract_first()
        except:
            item["userId"] = None

        return item