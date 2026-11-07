from datetime import datetime
import time

# Get current date and time
now = datetime.now()

# 1. Print only the time component (HH:MM:SS.microsecond)
print(now.time())  

# 2. Print individual parts
print(f"Hour: {now.hour}, Minute: {now.minute}, Second: {now.second}")

# 3. Format it cleanly as a string (e.g., 14:30:15)
formatted_time1 = now.strftime("%H:%M:%S")
# print("Formatted Time:", formatted_time)

time.sleep(5)
now = datetime.now()
formatted_time2 = now.strftime("%H:%M:%S")
# print("Formatted Time:", formatted_time)
time_difference = datetime.strptime(formatted_time2, "%H:%M:%S") - datetime.strptime(formatted_time1, "%H:%M:%S")
print("Time Difference:", time_difference)