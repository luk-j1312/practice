name = "Luke"
age = 20
GPA = 3.75
is_student = True
print("Name:", name, " Age:", age, " GPA:", GPA, " is Student:", is_student)
print(type(GPA))
print(type(is_student))

# lists --- ordered collections
prices = [10.99, 5.99, 3.50, 11.2, 7.99, 2.5]
print(prices)
print(prices[0])  # Accessing the first element
print(prices[1])  # Accessing the second element
print(prices[-1])  # Accessing the last element
print(len(prices))  # Length of the list
print(prices[1:3])  # Slicing the list
print(sum(prices))  # Sum of the list
print(max(prices))  # Maximum value in the list
print(min(prices))  # Minimum value in the list
for price in prices:
    print(f"price is {price}, doubled is {price*2}")
    if price > 10:
        print(f"{price} is high")
    elif price > 5:
        print(f"{price} is mid-range")
    else:
        print(f"{price} is low")

# list comprehension
high_prices = [price for price in prices if price > 10]
print(high_prices)

# functions
def describe_price(price):
    if price > 10:
        return f"{price} is high"
    elif price > 5:
        return f"{price} is mid-range"
    else:
        return f"{price} is low"
print(describe_price(12.5))
print(describe_price(7.5))
print(describe_price(3.0))

def summarize_prices(prices):
    average = sum(prices)/len(prices)
    return {
        "average": average,
        "max": max(prices),
        "min": min(prices)
    }
summary = summarize_prices(prices)
print(summary["average"])
print(summary)

