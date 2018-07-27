#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author: Dongyouyuan
# @Software: PyCharm
# @File: googleMap.py
# @Time: 17-8-2 上午11:31

from googleplaces import GooglePlaces
import googlemaps

import sys

reload(sys)
sys.setdefaultencoding('utf8')

class GoogleMaps(object):
    """提供google maps服务"""

    def __init__(self):

        self._GOOGLE_MAPS_KEY = "AIzaSyDQJJP2Tm90sMgY6RR6q8D9zaHNtamR3UI"
        self._Google_Places = GooglePlaces(self._GOOGLE_MAPS_KEY)
        self._Google_Geocod = googlemaps.Client(key=self._GOOGLE_MAPS_KEY)

    def _text_search(self, query, language=None, location=None):
        """
        根据搜索字符串,请求google API传回推荐的列表
        :param query: 搜索字符串
        :param language: 语言
        :param location: 地区筛选
        :return:
        """
        # lat_lng = {"lat": "22.5745761", "lng": "113.9393772"}
        # 经纬度附近搜索
        # text_query_result = self.self._Google_Places.text_search(query='Gong Yuan', lat_lng=lat_lng)
        # location 为人可认识的名称
        # text_query_result = self.self._Google_Places.text_search(query='Tang Lang Shan', location='shenzhen')
        # 指定语言搜索
        text_query_result = self._Google_Places.text_search(query=query, language=language, location=location)
        return text_query_result.places

    def _reverse_geocode(self, lat, lng, language=None):
        """
        根据经纬度请求google API获取坐标信息,返回信息
        :param lat: 纬度
        :param lng:经度
        :param language:语言
        :return:
        """
        # 根据经纬度获取地址信息 pincode
        list_reverse_geocode_result = self._Google_Geocod.reverse_geocode((lat, lng), language=language)
        # print json.dumps(list_reverse_geocode_result, indent=4)
        return list_reverse_geocode_result

    def _return_reverse_geocode_info(self, lat, lng, language=None):
        """
        整理信息
        :param lat:纬度
        :param lng:经度
        :param language:语言
        :return:
        """
        list_reverse_geocode = self._reverse_geocode(lat, lng, language=language)
        if list_reverse_geocode:
            city = ''
            pincode = ''
            route = ''
            neighborhood = ''
            sublocality = ''
            administrative_area_level_1 = ''
            country = ''
            street_number = ''
            # 全名地址
            formatted_address = list_reverse_geocode[0]['formatted_address']
            for address_info in list_reverse_geocode[0]['address_components']:
                # 城市标识为locality
                if 'locality' in address_info['types']:
                    city = address_info['long_name']
                # 邮政编码标识为postal_code
                elif 'postal_code' in address_info['types']:
                    pincode = address_info['long_name']
                # 街道路
                elif 'route' in address_info['types']:
                    route = address_info['long_name']
                # 相似地址名
                elif 'neighborhood' in address_info['types']:
                    neighborhood = address_info['long_name']
                # 地区名
                elif 'sublocality' in address_info['types']:
                    sublocality = address_info['long_name']
                # 省份
                elif 'administrative_area_level_1' in address_info['types']:
                    administrative_area_level_1 = address_info['long_name']
                # 国家
                elif 'country' in address_info['types']:
                    country = address_info['long_name']
                # 门牌号
                elif 'street_number' in address_info['types']:
                    street_number = address_info['long_name']
            return {'city': city, 'pincode': pincode, 'route': route, 'neighborhood': neighborhood,
                    'sublocality': sublocality, 'administrative_area_level_1': administrative_area_level_1,
                    'country': country, 'formatted_address': formatted_address, 'street_number': street_number}
        else:
            return None

    def get_formatted_address(self, lat, lng, language=None):
        list_reverse_geocode = self._reverse_geocode(lat, lng, language=language)
        loc_num = len(list_reverse_geocode)
        address = []
        for i in range(loc_num):
            address.append(list_reverse_geocode[i]['formatted_address'])
        return address

    def get_neighborhood(self, lat, lng, language=None):
        list_reverse_geocode = self._reverse_geocode(lat, lng, language=language)
        loc_num = len(list_reverse_geocode)
        neighbor = []
        for i in range(loc_num):
            for address_info in list_reverse_geocode[i]['address_components']:
                if 'neighborhood' in address_info['types']:
                    neighbor.append(address_info['long_name'])
        return neighbor          


    def get_pincode_city(self, lat, lng, language=None):
        """
        根据经纬度获取该地区详细信息
        :param lat: 纬度
        :param lng: 经度
        :return:
        """
        reverse_geocode_info = self._return_reverse_geocode_info(lat, lng, language=language)
        if reverse_geocode_info:
            return {'city': reverse_geocode_info['city'], 'pincode': reverse_geocode_info['pincode']}
        else:
            return None

    def get_address_recommendation(self, query, language=None, location=None):
        """
        获取输入地址的推荐地址(最多返回5个)
        :param query: 搜索地址名称
        :param language: 语言
        :param location: 地区筛选
        :return:
        """
        return_size = 5
        list_return_info = list()
        list_places_text_search_result = self._text_search(query=query, language=language, location=location)
        # 默认返回5条数据
        if len(list_places_text_search_result) > return_size:
            list_places_text_search_result = list_places_text_search_result[:return_size]
        for place in list_places_text_search_result:
            result_geocode = self._return_reverse_geocode_info(place.geo_location['lat'], place.geo_location['lng'], language=language)
            # 数据不为空
            if result_geocode:
                # 地点全路径加上地点名
                result_geocode['formatted_address'] = '{} {}'.format(place.name, result_geocode['formatted_address'])
                result_geocode['place_name'] = place.name
                # 经纬度
                result_geocode['lat'] = '{}'.format(place.geo_location['lat'])
                result_geocode['lng'] = '{}'.format(place.geo_location['lng'])
                list_return_info.append(result_geocode)
        return list_return_info


if __name__ == '__main__':
    # 使用实例
    import json
    google_maps = GoogleMaps()
    # result_recommendation_list = google_maps.get_address_recommendation(query='天河公园', language='zh', location='中国')
    # pincode_city = google_maps.get_pincode_city(22.547143, 114.062753)
    # print json.dumps(result_recommendation_list, indent=4, encoding='utf-8')
    # print '*'*100
    # print pincode_city
    # city_info = google_maps._return_reverse_geocode_info(33.9424, -118.4137)
    # city_info = google_maps._return_reverse_geocode_info(34.0928, -118.3287)  
    city_info = google_maps._return_reverse_geocode_info(39.987642, 116.333374)                          
    # print(city_info)
    city_neighbor = google_maps.get_formatted_address(34.0928, -118.3287)
    print(city_neighbor)