import requests
from datetime import datetime

#import json

# get data for each instance of listed properties
def fetch_property_data_for_id(id):
    print('Searching for property with id {}'.format(id))
    payload = {
        'year': '2018', 
        'card': '',
        'parcelid': id, # ID is a dinamic parameter 
        'type': 'property_card'
    }
    r = requests.post('https://property.spatialest.com/nc/mecklenburg/data/propertycard', data=payload)
    
    # get a JSON-object
    json_data = r.json()
    
    
    # fetch attributes for each single property
    id = json_data['ParcelOverview'][0]['parcelidentifier'] if json_data.get('ParcelOverview', None) else None
    neighborhood = json_data['ParcelOverview'][0]['neighborhoodidentifier'] if json_data.get('ParcelOverview', None) else None
    owner1 = json_data['ParcelOverview'][0]['Owner1'] if json_data.get('ParcelOverview', None) else None
    owner2 = json_data['ParcelOverview'][0]['Owner2'] if json_data.get('ParcelOverview', None) else None
    address = json_data['ParcelOverview'][0]['location_address'] if json_data.get('ParcelOverview', None) else None
    value_property = json_data['ParcelOverview'][0]['TotalAssessedValue'] if json_data.get('ParcelOverview', None) else None
    year_built = json_data['residential'][0]['yearbuilt'] if json_data.get('residential', None) else None
    building_type = json_data['residential'][0]['buildingtype'] if json_data.get('residential', None) else None
    num_fireplaces = json_data['residential'][0]['Fireplaces'] if json_data.get('residential', None) else None
    num_bath = json_data['residential'][0]['FullBath'] if json_data.get('residential', None) else None
    area = json_data['residential'][0]['totalarea'] if json_data.get('residential', None) else None
    
    formatted_sale_date = json_data['ParcelOverview'][0]['SaleDate'] if json_data.get('ParcelOverview', None) else None
    sale_date = datetime.strptime(formatted_sale_date, '%m/%d/%Y').date() if formatted_sale_date != None else None                     

    heat = json_data['residential'][0]['heat'] if json_data.get('residential', None) else None
    heat_fuel = json_data['residential'][0]['heatfuel'] if json_data.get('residential', None) else None
    foundation = json_data['residential'][0]['foundation'] if json_data.get('residential', None) else None
    num_half_bath = json_data['residential'][0]['HalfBath'] if json_data.get('residential', None) else None
    story = json_data['residential'][0]['storyheight'] if json_data.get('residential', None) else None
    ext_wall = json_data['residential'][0]['extwall'] if json_data.get('residential', None) else None
    land_value = json_data['ParcelOverview'][0]['TotalLandValue'] if json_data.get('ParcelOverview', None) else None
    building_value = json_data['ParcelOverview'][0]['TotalBuildingValue'] if json_data.get('ParcelOverview', None) else None
    features_value = json_data['ParcelOverview'][0]['TotalYardItemValue'] if json_data.get('ParcelOverview', None) else None
    
    # create a dictionary with data fetched
    data = {
        'id' : id,
        'neighborhood' : neighborhood,
        'owner_1' : owner1,
        'owner_2' : owner2,
        'address' : address,
        'year_built': year_built,
        'building_type' : building_type,
        'num_fireplaces' : num_fireplaces, 
        'num_bath' : num_bath,
        'area' : area,
        'value_property' : value_property,
        'json_blob' : json_data,
        
        'sale_date' : sale_date,
        'heat' : heat,
        'heat_fuel' : heat_fuel,
        'foundation' : foundation,
        'num_half_bath' : num_half_bath,
        'story' : story,
        'ext_wall' : ext_wall,
        'land_value' : land_value,
        'building_value' : building_value,
        'features_value' : features_value
    }
    
    return data


# go over the pages
def search(page=1):
    print("Searching page '{}'".format(page))
    payload = {
        'type': 'search', 
        'term': 'R100_luc',
        'page': page # page is a dinamic parameter
    }
    r = requests.post('https://property.spatialest.com/nc/mecklenburg/data/search', data=payload)
    search_result_json = r.json() # create a dictionary of JSON-text response
    
    
    # start collecting property data
    property_data_this_page = []

    for searchResult in search_result_json['searchResults']:
        id = searchResult['PropertyID']
        property_data_this_page.append(fetch_property_data_for_id(id))
        
    # use recursion to get property data for remaining pages
    num_search_pages=len(search_result_json['pages'])            
    if page < num_search_pages:     
        property_data_remaining_pages = search(page+1)
        # merge into single result
        property_data_this_page = property_data_this_page + property_data_remaining_pages
        
    return property_data_this_page
    

# test for debug purposes
#search_result = fetch_property_data_for_id(20153)
#print(json.dumps(search_result, indent=4, sort_keys=True, default=json_util.default))

# test for debug purposes
#search_result = search(39)
#print(json.dumps(search_result, indent=4, sort_keys=True))

