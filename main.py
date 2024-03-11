from flask import Flask, render_template
import csv
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Define the paths
csv_file_path = "Data/data_preprocessed (1).csv"
output_path = "Data/modified_data.csv"

# Function to read data from CSV file
def read_data_from_csv():
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data
  
# Read data from CSV file
data = read_data_from_csv()
#Condense to a normal week of the season 16 games
data = data[0:8]

line1 = []
line2 = []

#edit the data so we only get rows we need
for row in data:
  team_home = row["team_home"]
  score_home = row["score_home"]
  team_home_current_win_pct = row["team_home_current_win_pct"]
  over_under_line1 = row["over_under_line"]
  spread_favorite1 = row["spread_favorite"]
  line1.append((team_home, score_home, team_home_current_win_pct, over_under_line1, spread_favorite1))

  team_away = row["team_away"]
  score_away = row["score_away"]
  team_away_current_win_pct = row["team_away_current_win_pct"]
  over_under_line2 = row["over_under_line"]
  spread_favorite2 = float(row["spread_favorite"])*-1
  line2.append((team_away, score_away, team_away_current_win_pct, over_under_line2, spread_favorite2))

with open(output_path, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)

  # Write headers
  writer.writerow(['Team', 'Score', 'Current Win Percentage', 'Over/Under Line', 'Spread Favorite'])
  # Write line1 values
  for values in line1:
      writer.writerow(values)

  # Write line2 values
  for values in line2:
      writer.writerow(values)



if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)
