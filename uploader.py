import json
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from cookies import chrome_cookies
from seleniumwire.undetected_chromedriver.v2 import Chrome, ChromeOptions

# chrome_options.add_argument("--headless")
from time import sleep
class Uploader():

	"""

	Uploader object which would handle all the upload functions
	as well as browser automation via Selenium


	"""
	def init_video(self,filepath):

		"""

		A function to read below parameters from json file:
		Title *Required
		Description,
		Tags,
		Category,
		Age Restrictions

		Which are held in a json file. Initiate an object with passing the 
		file path as the parameter.

		"""

		# If only the filename is given, add the extension
		if not ('.json' in filepath):
			filepath+='.json'
		data = json.load(open(filepath,'r'))
		return data

	def __init__(self,visible=False):
		# Initialize the browser object
		# chrome_options = Options()
		# chrome_options.add_argument("--disable-gpu")
		# if visible:
		# 	chrome_options.add_argument("--headless")
		# 	chrome_options.add_argument("--window-size=1280,800")
		# self.browser = webdriver.Chrome(executable_path="/home/rishirules/Downloads/chromedriver", chrome_options=chrome_options)
		options = {}

		chrome_options = ChromeOptions()
		chrome_options.add_argument('--user-data-dir=hash')
		chrome_options.add_argument("--disable-gpu")
		chrome_options.add_argument("--incognito")
		chrome_options.add_argument("--disable-dev-shm-usage")
		self.browser= Chrome(executable_path="/home/rishirules/Downloads/chromedriver", seleniumwire_options={}, options=chrome_options) 

		self.browser.get('https://stackoverflow.com/users/signup?ssrc=head&returnurl=%2fusers%2fstory%2fcurrent%27')
		sleep(3)
		self.browser.find_element_by_xpath('//*[@id="openid-buttons"]/button[1]').click()
		self.browser.find_element_by_xpath('//input[@type="email"]').send_keys("deepstyle42")
		self.browser.find_element_by_xpath('//*[@id="identifierNext"]').click()
		sleep(3)
		self.browser.find_element_by_xpath('//input[@type="password"]').send_keys("Raulrishi1")
		self.browser.find_element_by_xpath('//*[@id="passwordNext"]').click()
		sleep(2)

	def get_cookies(self):

		"""

		Gets cookies from user's default chrome browser,
		User has to be logged in to YouTube in default Chrome Browser
		for this function to return cookies.

		"""
		self.cookies = chrome_cookies('https://www.youtube.com')
		return cookies

	def inject_cookies(self):

		"""

		Inject cookies to the browser object

		"""
		# Go to youtube.com so that we can inject the cookies
		self.browser.get('https://www.youtube.com')

		# inject cookies to the Chrome Browser that is used by Selenium
		for c in cookies:
		# set expiry date to infinity or something
			c['expiry'] = 33333333333
			self.browser.add_cookie(c)

	def upload_video(self,filepath):

		"""

		Gets video data, then uploads it by automating clicks and actions on Selenium
		WebDriver object.

		"""
		# read video data
		data = self.init_video(filepath)
		# if title is not in data, return immediately
		if 'title' not in data.keys():
			raise Exception('Field Required (title and video_path are required)')
		# navigate to the upload page
		self.browser.get('https://studio.youtube.com/')
		sleep(2)
		# # click on 'Create' button
		# self.browser.find_element_by_id('create-icon').click()
		# sleep(2)
		# Click on 'Upload Video'
		self.browser.find_element_by_id('upload-icon').click()
		sleep(2)
		# Get Input Element
		self.browser.find_element_by_css_selector('input[name="Filedata"]').send_keys(data['video_path'])
		sleep(2)
		# # Title
		# self.browser.find_element_by_id('textbox').send_keys(data.title)
		# sleep(2)
		# # Desc
		# self.browser.find_element_by_class_name('description-textarea').find_element_by_css_selector('div#textbox').send_keys(data.desc)
		# sleep(2)

		# ## TODO: Add Other functionalities here!
		# ## TODO: Add info about video upload here!
		nokidsid="ink"
		self.browser.find_element_by_xpath('//button[normalize-space()="No, it\'s not made for kids"]').click()
		sleep(2)
		# Next
		self.browser.find_element_by_id('next-button').click()
		self.browser.find_element_by_id('next-button').click()
		sleep(2)
		visibilities = ['PUBLIC','UNLISTED','PRIVATE']
		self.browser.find_element_by_css_selector('paper-radio-button[name="'+visibilities[0]+'"]').click()  
		sleep(2)
		# Done
		self.browser.find_element_by_id('done-button').click()
