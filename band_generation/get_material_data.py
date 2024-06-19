from mp_api.client import MPRester
import numpy as np
import matplotlib.pyplot as plt
import csv



with open('material_data.csv','w') as write_file:
    data_writer = csv.writer(write_file)
    
    with open('table_export.csv','r') as read_file:
        data_reader = csv.reader(read_file)
        
        with MPRester(api_key="SFTntPIf0QnTef7f6sZKkBSwGILrxlXZ") as mpr:
        bandstructure = mpr.get_bandstructure_by_material_id("mp-1095269")

        plt.figure()
        for element in bandstructure.bands:
            for i in range(bandstructure.nb_bands):
                band = list(bandstructure.bands.values())[0][i] - bandstructure.efermi
                krange = np.linspace(-np.pi, np.pi, len(band))
                poly_coeffs = np.polyfit(krange, band, 20)
                data_writer.writerow(poly_coeffs)
                plt.plot(np.linspace(-np.pi, np.pi, len(band)), band)

plt.xlabel("k"); plt.ylabel("E"); plt.title("band structure")
plt.show()