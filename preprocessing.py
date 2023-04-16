import pandas as pd
import json
# df = pd.read_json('./recipes_full.json')

with open("./recipes_full.json", "r") as read_file:
    data = json.load(read_file)
    arr = []
    # dropping unneeded keys adn metadata and keep only reciep related fields
    for key in data.keys():
        crn_rec = data[key]
        crn_rec['id'] = key
        del crn_rec['ph mash']
        del crn_rec['url']
        del crn_rec['color']
        del crn_rec['num rating']
        del crn_rec['rating']
        del crn_rec['views']
        del crn_rec['abv']
        del crn_rec['ibu']
        del crn_rec['og']
        del crn_rec['fg']
        del crn_rec['hops Summary']
        arr.append(crn_rec)
    
    with open("./recipes_processed.json", "w") as final:
        json.dump(arr, final, indent=6)

# print(df.head)