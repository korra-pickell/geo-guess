# PRIMARY DATA SCRAPING SCRIPT

# Logs both location and related images from the Google Maps engine
import time, os
import numpy as np
import cv2
import pyautogui as pog
import matplotlib.pyplot as plt
from mss import base, mss
from selenium import webdriver
from geopy.geocoders import Nominatim
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

MAX_CYCLES = 100000
pog.PAUSE = 0.3

def launch_browser():
    binary = FirefoxBinary(r'C:\Program Files\Mozilla Firefox\firefox.exe')
    browser = webdriver.Firefox(firefox_binary=binary)

    return browser

def clean_ui(browser):
    # Remove Bottom Banner
    browser.execute_script("""
    return document.getElementById("bottom-box").remove();
    """)

def ui_settings():
    pog.click(290,140)
    pog.click(290,700)
    time.sleep(0.2)
    pog.click(290,140)

def get_address(browser):
    address = browser.find_element(By.ID, 'address').text
    return address

def get_coords(address):
    geolocator = Nominatim(user_agent="my_request")
    try:
        location = geolocator.geocode(address)
    except:
        print('----------------------------------------- EXCEPTION')
        return None
    if location is None:
        return None
    else:
        return (location.latitude, location.longitude)

def next_scene():
    pog.click(450,140)

def grab_image():
    sct = mss()
    bbox = {'top': 198, 'left': 10, 'width': 1225, 'height': 745}
    current_frame = np.array(sct.grab(bbox))
    #current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    return current_frame

def save_data(coords,image):
    coord_dir = r'E:\DATA\GEO-GUESS\coordinates'
    coord_file = open(coord_dir+'\\coordinates.txt','r+')
    coord_file_lines = coord_file.readlines()
    current_num = len(coord_file_lines)

    coord_file.write('E:\\DATA\\GEO-GUESS\\images\\' + str(current_num) + '.jpg' + ',' + str(coords[0]) + ',' + str(coords[1]) +'\n')
    coord_file.close()
    cv2.imwrite('E:\\DATA\\GEO-GUESS\\images\\' + str(current_num) + '.jpg', image)

if __name__ == '__main__':
    browser = launch_browser()
    browser.get(r'https://www.mapcrunch.com/')
    clean_ui(browser)
    ui_settings()
    next_scene()
    time.sleep(1)
    for num in range(MAX_CYCLES):
        address = get_address(browser)
        print("ADDRESS: " + str(address))
        coords = get_coords(address)
        if coords is not None:
            image = grab_image()
            save_data(coords,image)
        next_scene()
        time.sleep(3)
