import ee
import requests
import os
import datetime
from Coconut_Locations import locations

# Initialize
ee.Initialize(project='skepter2k')

current_year = datetime.datetime.now().year

# Sentinel-2 collection
sentinel = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterDate('2018-01-01', f'{current_year}-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .filter(ee.Filter.lt('MEAN_SOLAR_ZENITH_ANGLE', 80))

def mask_clouds(image):
    """Mask clouds & shadows using SCL band"""
    scl = image.select('SCL')
    cloud_shadow = scl.eq(3)
    clouds = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
    mask = clouds.Or(cloud_shadow).Not()
    return image.updateMask(mask)

def usable_fraction(image, roi):
    """Calculate fraction of usable (non-cloud) pixels in ROI"""
    scl = image.select('SCL')
    cloud_shadow = scl.eq(3)
    clouds = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
    mask = clouds.Or(cloud_shadow).Not()
    stats = mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    return stats.get('SCL')

output_dir = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_Coconut_Ad"
os.makedirs(output_dir, exist_ok=True)

log_file = os.path.join(output_dir, "gee_export_log.txt")

with open(log_file, "w") as log:
    log.write("Download Log:\n" + "=" * 50 + "\n")

    for city, points in locations.items():
        if not isinstance(points, list):
            points = [points]

        for idx, center in enumerate(points):
            roi = center.buffer(500).bounds()

            for i in range((current_year - 2018 + 1) * 12):
                year = (i // 12) + 2018
                month = (i % 12) + 1
                start = ee.Date.fromYMD(year, month, 1)
                end = start.advance(1, 'month')

                filtered = sentinel.filterBounds(roi).filterDate(start, end)
                count = filtered.size().getInfo()

                log_message = f"{city} ({year}-{month}): {count} images available"
                print(log_message)
                log.write(log_message + "\n")

                if count == 0:
                    continue

                # Least cloudy image
                best_image = filtered.sort('CLOUDY_PIXEL_PERCENTAGE').first()

                try:
                    fraction = usable_fraction(best_image, roi).getInfo()
                except Exception as e:
                    error_msg = f"Error computing usable fraction for {city} {year}-{month} (P{idx+1}): {e}"
                    print(error_msg)
                    log.write(error_msg + "\n")
                    continue

                if fraction is None or fraction < 1:  # 70% usable pixels
                    skip_msg = f"Skipped {city} {year}-{month} (P{idx+1}) - usable fraction {fraction}"
                    print(skip_msg)
                    log.write(skip_msg + "\n")
                    continue

                # Mask clouds
                masked_image = mask_clouds(best_image)

                # Export all bands B1-B12 + SCL
                all_bands = [
                    'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','SCL'
                ]

                img_export = masked_image.select(all_bands).toFloat()

                description = f"S2_{city}_BestImage_{year}_{month}_P{idx+1}"

                try:
                    url = img_export.getDownloadURL({
                        'scale': 10,
                        'region': roi.getInfo(),
                        'format': 'GEO_TIFF',
                        'name': description
                    })

                    response = requests.get(url)
                    if response.status_code == 200:
                        file_path = os.path.join(output_dir, f"{description}.tif")
                        with open(file_path, 'wb') as f:
                            f.write(response.content)

                        success_msg = f"Successfully downloaded {file_path}"
                        print(success_msg)
                        log.write(success_msg + "\n")
                    else:
                        error_msg = f"Failed to download {description}. Status code: {response.status_code}"
                        print(error_msg)
                        log.write(error_msg + "\n")

                except Exception as e:
                    error_msg = f"Error downloading {description}: {str(e)}"
                    print(error_msg)
                    log.write(error_msg + "\n")

print(f"\nAll downloads attempted. Check {log_file} for details.")
