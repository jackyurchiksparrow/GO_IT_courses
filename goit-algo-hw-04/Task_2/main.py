from data_processing.process import get_cats_info


if __name__ == "__main__":
    cats_info = get_cats_info("raw_data/cats_file.txt")
    
    print(cats_info)
