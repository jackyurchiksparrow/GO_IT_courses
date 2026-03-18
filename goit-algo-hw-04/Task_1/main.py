from data_processing.process import total_salary


if __name__ == "__main__":
    total, average = total_salary("raw_data/salary_file.txt")
    print(f"Загальна сума заробітної плати: {total}, Середня заробітна плата: {average}")