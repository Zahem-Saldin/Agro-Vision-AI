import xml.etree.ElementTree as ET
import csv

kml_file_path = r"C:\Users\Zahem Saldin\Desktop\Locations\Coconut.kml"
csv_output_path = r"C:\Users\Zahem Saldin\Desktop\Locations\Coconut_Coordinates.csv"
py_dict_output_path = r"C:\Users\Zahem Saldin\Desktop\Locations\Coconut_Coordinates.py"

# Parse KML
tree = ET.parse(kml_file_path)
root = tree.getroot()

# Handle namespaces
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

all_coords = []

# Find all Placemark coordinates
for placemark in root.findall('.//kml:Placemark', ns):
    name_elem = placemark.find('kml:name', ns)
    name = name_elem.text if name_elem is not None else None
    for coord_elem in placemark.findall('.//kml:coordinates', ns):
        coord_text = coord_elem.text.strip()
        # Coordinates can be multiple, separated by spaces
        for c in coord_text.split():
            lon, lat, *rest = map(float, c.split(','))
            all_coords.append((lon, lat, name))

# Save to CSV
with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Longitude', 'Latitude', 'Name'])
    for lon, lat, name in all_coords:
        writer.writerow([lon, lat, name or 'Unnamed'])

print(f"Saved {len(all_coords)} coordinates to CSV: {csv_output_path}")

# Save Python dictionary for GEE
dict_content = "locations = {\n"
for idx, (lon, lat, name) in enumerate(all_coords):
    safe_name = name.replace("'", "_").replace(" ", "_") if name else f"Location_{idx+1}"
    dict_content += f"    '{safe_name}': ee.Geometry.Point({lon}, {lat}),\n"
dict_content += "}\n"

with open(py_dict_output_path, 'w', encoding='utf-8') as pyfile:
    pyfile.write(dict_content)

print(f"Saved Python dictionary for GEE to: {py_dict_output_path}")
