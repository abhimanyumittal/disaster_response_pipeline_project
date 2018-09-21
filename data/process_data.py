import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv file and merge them in a single dataframe
    Args:
      messages_filepath: path of the messages csv file
      categories_filepath: path of the categories csv file
    Returns:
      df: merged dataframe of messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge messages and categories in a single dataframe
    df = pd.merge(messages, categories, left_on='id', right_on='id')

    return df


def process_categories(df):
    """
    Return dataframe categories
    Args:
      df: Dataframe
    Returns:
      categories: categories dataframe
    """
    # split categories column and create categories dataframe
    categories = df.categories.str.split(';', expand=True)

    # select first row of the dataframe
    row = categories.iloc[0]

    # split the values and get the string until second last character
    category_colnames = lambda x: [str(y)[:-2] for y in x]
    # assign categories dataframe column names
    categories.columns = category_colnames(row)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1] if int(x[-1]) < 2 else 1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(str).astype(int)

    return categories


def clean_data(df):
  """
  Clean the dataset
  Args:
    df: merged dataframe
  Returns:
    df: cleaned dataframe
  """
  # get categories in a seperate dataframe
  categories = process_categories(df)
  # remove categories column
  df = df.drop(['categories'],axis=1)
  # concat merged dataframe (without categories column) and categories dataframe
  df = pd.concat([df,categories],axis=1)

  # for checking of number of duplicates, can use
  # print(df.duplicated().sum())

  # remove duplicate rows
  df = df.drop_duplicates()

  return df


def save_data(df, database_filename):
  """
  save dataframe to db
  Args:
    df: merged dataframe
    database_filename: name of database filename
  """
  engine = create_engine('sqlite:///'+database_filename)
  df.to_sql('disaster_response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[
            1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
