import os
import time
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

from pprint import pprint
from zapv2 import ZAPv2 as ZAP
from selenium.webdriver import FirefoxProfile

# Create options object
opts = Options()

# Set the options to send traffic through 127.0.0.1:8080 where ZAP listens
opts.set_preference('network.proxy.type', 1)
opts.set_preference('network.proxy.http', '127.0.0.1')
opts.set_preference('network.proxy.http_port', 8080)
opts.set_preference('network.proxy.ssl', '127.0.0.1')
opts.set_preference('network.proxy.ssl_port', 8080)

# Set the options to headless mode
opts.add_argument('--headless')

# Initialize a new driver with all the proxy and headless options
driver = Firefox(options=opts)
target = os.environ['TARGET']

# Run the selenium tests
# As the below tests run, the webdriver driver will be proxied through ZAP
# ZAP can later use the below URLs, and perform further spidering on them
try:
    driver.get(target + '/taskManager/register/')
    time.sleep(2)
    # enter the username, password, firstname,lastname etc.
    driver.find_element(By.ID, 'id_username').send_keys('user10')
    time.sleep(1)
    driver.find_element(By.ID, 'id_first_name').send_keys('user')
    time.sleep(1)
    driver.find_element(By.ID, 'id_last_name').send_keys('user')
    time.sleep(1)
    driver.find_element(By.ID, 'id_email').send_keys('user@user1.com')
    time.sleep(1)
    driver.find_element(By.ID, 'id_password').send_keys('user123')
    time.sleep(1)
    submit = driver.find_element("css selector", '.btn.btn-danger').click()
    time.sleep(2)
    # login
    driver.get(target + '/taskManager/login/')
    driver.find_element(By.ID, 'username').send_keys('user10')
    driver.find_element(By.NAME, "password").send_keys('user123')
    submit = driver.find_element(By.XPATH, "//button[@type='submit']")
    submit.click()
    time.sleep(2)
    # spider the URL's
    driver.get(target + "/taskManager/dashboard/")
    time.sleep(2)
    driver.get(target + "/taskManager/task_list/")
    time.sleep(2)
    driver.get(target + "/taskManager/project_list/")
    time.sleep(2)
    driver.get(target + "/taskManager/search/")
    time.sleep(2)
    # editing some data for better scan results
    driver.get(target + "/taskManager/profile/")
    time.sleep(5)
    driver.find_element(By.NAME, "first_name").send_keys('firstroot')
    driver.find_element(By.NAME, 'last_name').send_keys('lastroot')
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(2)
finally:
    driver.close()

# Initialize ZAP client
print("Initializing ZAP python client")
zap = ZAP(proxies={'http': 'http://127.0.0.1:8080', 'https':'https://127.0.0.1:8080'})

# Start traditional spider
print('Starting traditional spider')
scanid = zap.spider.scan(target)

time.sleep(2)
while(int(zap.spider.status(scanid))<100):
    print('Traditional spider progress %: {}'.format(zap.spider.status(scanid)))
    time.sleep(5)

traditionalspiderresults = zap.spider.results(scanid)

# Printing traditional spider results
print('\n'.join(map(str, traditionalspiderresults)))

# Start AJAX spider
print('Starting the ZAP AJAX spider')
scanid = zap.ajaxSpider.scan(target)

time.sleep(2)
while zap.ajaxSpider.status == 'running':
    if time.time() > timeout:
        break
    print('Ajax Spider status' + zap.ajaxSpider.status)
    time.sleep(2)

ajaxspiderresults = zap.ajaxSpider.results(start=0, count=100)

# Printing ajax spider results
print('\n'.join(map(str, ajaxspiderresults)))

# Print all the identified URLs
print('Printing all identified URLs')
print('\n'.join(map(str, traditionalspiderresults)))
print('\n'.join(map(str, ajaxspiderresults)))