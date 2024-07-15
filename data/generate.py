if __name__ == '__main__':
    import json

    fake_data = [{"x": i, "y": i * 2 + 3} for i in range(100)]

    with open("data.json", "w") as f:
        json.dump(fake_data, f, indent=4)
