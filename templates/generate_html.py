import csv

# Read data from CSV file
data = []
with open('data/your_dataset.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Generate HTML table rows
html_rows = ""
for row in data:
    html_rows += f"""
        <tr>
            <td>{row['Team']}</td>
            <td class="numbers">{row['Score']}</td>
            <td class="numbers">{row['Current Win Percentage']}</td>
            <td class="numbers">{row['Over/Under Line']}</td>
            <td class="numbers">{row['Spread Favorite']}</td>
            <td class="win-percentage">{row['Win%']}</td>
            <td class="numbers">$0</td>
        </tr>
    """

# Write the HTML rows to a file
with open('output.html', 'w') as file:
    file.write(html_rows)