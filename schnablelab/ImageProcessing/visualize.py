import csv
import matplotlib.pyplot as plt
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path of the csv file")
args = vars(ap.parse_args())
csv_path = args["path"]
plant_ID = ((csv_path.split('/')[-1]).split('_')[-1]).split('.')[0]

traits = []
date = []
inflorescence_width = []
inflorescence_height = []
stem_height = []
plant_height = []
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if(len(row) < 6):
            continue
        if line_count == 0:
            traits = row
            line_count += 1
        else:
            date.append(row[1])
            inflorescence_width.append(eval(row[2]))
            inflorescence_height.append(eval(row[3]))
            stem_height.append(eval(row[4]))
            plant_height.append(eval(row[5]))
            line_count += 1


plt.figure(figsize = (30,30))
plt.subplot(4, 1, 1)
plt.title(plant_ID+'_inflorescence_width',fontsize=24)
plt.xlabel('date',fontsize=14)
plt.ylabel('inflorescence_width in cm',fontsize=14)
plt.plot(date, inflorescence_width,color="r",marker="*",label="inflorescence_width")
for a,b in zip(date,inflorescence_width):
    plt.text(a,b+0.05,'%.3f'%b,ha = 'center',va = 'bottom',fontsize=10)
plt.legend()


plt.subplot(4, 1, 2)
plt.title(plant_ID+'_inflorescence_height',fontsize=24)
plt.xlabel('date',fontsize=14)
plt.ylabel('inflorescence_height in cm',fontsize=14)
plt.plot(date, inflorescence_height,color="b",marker="*",label="inflorescence_height")
for a,b in zip(date,inflorescence_height):
    plt.text(a,b+0.05,'%.3f'%b,ha = 'center',va = 'bottom',fontsize=10)
plt.legend()

plt.subplot(4, 1, 3)
plt.title(plant_ID+'_stem_height',fontsize=24)
plt.xlabel('date',fontsize=14)
plt.ylabel('stem_height in cm',fontsize=14)
plt.ylim((0,100))
plt.plot(date, stem_height,color="g",marker="*",label="stem_height")
for a,b in zip(date,stem_height):
    plt.text(a,b+0.05,'%.3f'%b,ha = 'center',va = 'bottom',fontsize=10)
plt.legend()


plt.subplot(4, 1, 4)
plt.title(plant_ID+'_plant_height',fontsize=24)
plt.xlabel('date',fontsize=14)
plt.ylabel('plant_height in cm',fontsize=14)
plt.ylim((0,120))
plt.plot(date, plant_height,color="m",marker="*",label="plant_height")
for a,b in zip(date,plant_height):
    plt.text(a,b+0.05,'%.3f'%b,ha = 'center',va = 'bottom',fontsize=10)
plt.legend()

file_name = plant_ID+"_growth_curve.png"
plt.savefig(file_name)
#plt.show()
