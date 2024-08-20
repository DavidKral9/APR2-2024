from abc import abstractmethod, ABC
from json import load
from numbers import Real
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, Union, Any, List, Callable
from enum import Enum
from collections.abc import MutableSequence
import random
import math
class Type(Enum):
    Float = 0
    String = 1


def to_float(obj) -> float:
    """
    casts object to float with support of None objects (None is cast to None)
    """
    return float(obj) if obj is not None else None


def to_str(obj) -> str:
    """
    casts object to float with support of None objects (None is cast to None)
    """
    return str(obj) if obj is not None else None


def common(iterable): # from ChatGPT
    """
    returns True if all items of iterable are the same.
    :param iterable:
    :return:
    """
    try:
        # Nejprve zkusíme získat první prvek iterátoru
        iterator = iter(iterable)
        first_value = next(iterator)
    except StopIteration:
        # Vyvolá výjimku, pokud je iterátor prázdný
        raise ValueError("Iterable is empty")

    # Kontrola, zda jsou všechny další prvky stejné jako první prvek
    for value in iterator:
        if value != first_value:
            raise ValueError("Not all values are the same")

    # Vrací hodnotu, pokud všechny prvky jsou stejné
    return first_value


class Column(MutableSequence):# implement MutableSequence (some method are mixed from abc)
    """
    Representation of column of dataframe. Column has datatype: float columns contains
    only floats and None values, string columns contains strings and None values.
    """
    def __init__(self, data: Iterable, dtype: Type):
        self.dtype = dtype
        self._cast = to_float if self.dtype == Type.Float else to_str
        # cast function (it casts to floats for Float datatype or
        # to strings for String datattype)
        self._data = [self._cast(value) for value in data]

    def __len__(self) -> int:
        """
        Implementation of abstract base class `MutableSequence`.
        :return: number of rows
        """
        return len(self._data)

    def __getitem__(self, item: Union[int, slice]) -> Union[float,
                                    str, list[str], list[float]]:
        """
        Indexed getter (get value from index or sliced sublist for slice).
        Implementation of abstract base class `MutableSequence`.
        :param item: index or slice
        :return: item or list of items
        """
        return self._data[item]

    def __setitem__(self, key: Union[int, slice], value: Any) -> None:
        """
        Indexed setter (set value to index, or list to sliced column)
        Implementation of abstract base class `MutableSequence`.
        :param key: index or slice
        :param value: simple value or list of values

        """
        self._data[key] = self._cast(value)

    def append(self, item: Any) -> None:
        """
        Item is appended to column (value is cast to float or string if is not number).
        Implementation of abstract base class `MutableSequence`.
        :param item: appended value
        """
        self._data.append(self._cast(item))

    def insert(self, index: int, value: Any) -> None:
        """
        Item is inserted to colum at index `index` (value is cast to float or string if is not number).
        Implementation of abstract base class `MutableSequence`.
        :param index:  index of new item
        :param value:  inserted value
        :return:
        """
        self._data.insert(index, self._cast(value))

    def __delitem__(self, index: Union[int, slice]) -> None:
        """
        Remove item from index `index` or sublist defined by `slice`.
        :param index: index or slice
        """
        del self._data[index]

    def permute(self, indices: List[int]) -> 'Column':
        """
        Return new column which items are defined by list of indices (to original column).
        (eg. `Column(["a", "b", "c"]).permute([0,0,2])`
        returns  `Column(["a", "a", "c"])
        :param indices: list of indexes (ints between 0 and len(self) - 1)
        :return: new column
        """
        assert len(indices) == len(self)
        ...

    def copy(self) -> 'Column':
        """
        Return shallow copy of column.
        :return: new column with the same items
        """
        # FIXME: value is cast to the same type (minor optimisation problem)
        return Column(self._data, self.dtype)

    def get_formatted_item(self, index: int, *, width: int):
        """
        Auxiliary method for formating column items to string with `width`
        characters. Numbers (floats) are right aligned and strings left aligned.
        Nones are formatted as aligned "n/a".
        :param index: index of item
        :param width:  width
        :return:
        """
        assert width > 0
        if self._data[index] is None:
            if self.dtype == Type.Float:
                return "n/a".rjust(width)
            else:
                return "n/a".ljust(width)
        return format(self._data[index],
                      f"{width}s" if self.dtype == Type.String else f"-{width}.2g")

class DataFrame:
    """
    Dataframe with typed and named columns
    """
    def __init__(self, columns: Dict[str, Column]):
        """
        :param columns: columns of dataframe (key: name of dataframe),
                        lengths of all columns has to be the same
        """
        assert len(columns) > 0, "Dataframe without columns is not supported"
        self._size = common(len(column) for column in columns.values())
        # deep copy od dict `columns`
        self._columns = {name: column.copy() for name, column in columns.items()}

    def __getitem__(self, index: int) -> Tuple[Union[str,float]]:
        """
        Indexed getter returns row of dataframe as tuple
        :param index: index of row
        :return: tuple of items in row
        """
        ...

    def __iter__(self) -> Iterator[Tuple[Union[str, float]]]:
        """
        :return: iterator over rows of dataframe
        """
        for i in range(len(self)):
            yield tuple(c[i] for c in self._columns.values())

    def __len__(self) -> int:
        """
        :return: count of rows
        """
        return self._size
    def unique(self, col_name: str) -> 'DataFrame':
     """
    Returns a new DataFrame with duplicate rows removed based on the values in the specified column.
    
    :param col_name: The name of the column to check for duplicates.
    :return: A new DataFrame with duplicates removed.
     """
     if col_name not in self._columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    
     seen = set()
     unique_rows = []

     for i in range(len(self)):
        value = self._columns[col_name][i]
        if value not in seen:
            seen.add(value)
            unique_rows.append(tuple(c[i] for c in self._columns.values()))
    
    # Create new columns for the unique DataFrame
     new_columns = {
        name: Column([row[idx] for row in unique_rows], column.dtype)
        for idx, (name, column) in enumerate(self._columns.items())
    }
    
     return DataFrame(new_columns)
    def sample(self, n: int) -> 'DataFrame':
        """
        Returns a new DataFrame containing a random sample of rows from the original DataFrame.
        
        :param n: The number of rows to sample.
        :return: A new DataFrame with the sampled rows.
        """
        if n > len(self):
            raise ValueError("Sample size cannot be larger than the number of rows in the DataFrame.")
        
        sampled_indices = random.sample(range(len(self)), n)
        sampled_rows = [tuple(c[i] for c in self._columns.values()) for i in sampled_indices]
        
        # Create new columns for the sampled DataFrame
        new_columns = {
            name: Column([row[idx] for row in sampled_rows], column.dtype)
            for idx, (name, column) in enumerate(self._columns.items())
        }
        
        return DataFrame(new_columns)
    def product(self, axis: int = 0) -> 'DataFrame':
        """
        Returns a new DataFrame with the product of elements either along rows or columns.
        
        :param axis: 0 to multiply down the columns, 1 to multiply across the rows.
        :return: A new DataFrame with the products of the elements.
        """
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")
        
        if axis == 0:  # Multiply down the columns
            new_data = {
                name: Column([int(math.prod(column))], Type.Float)
                for name, column in self._columns.items()
            }
        elif axis == 1:  # Multiply across the rows
            products = []
            for i in range(len(self)):
                row_product = int(math.prod(self._columns[name][i] for name in self._columns))
                products.append(row_product)
            
            new_data = {
                "Product": Column(products, Type.Float)
            }
        
        return DataFrame(new_data)
    def replace(self, to_replace, value) -> 'DataFrame':
        """
        Returns a new DataFrame with all occurrences of `to_replace` replaced by `value`.
        
        :param to_replace: The value or list of values to be replaced.
        :param value: The value to replace `to_replace` with.
        :return: A new DataFrame with the replaced values.
        """
        if not isinstance(to_replace, list):
            to_replace = [to_replace]
    
        new_columns = {}
    
        for name, column in self._columns.items():
            new_data = [
                value if item in to_replace else item
                for item in column
            ]
            new_columns[name] = Column(new_data, column.dtype)
        
        return DataFrame(new_columns)
    def melt(self, id_vars: list, value_vars: list) -> 'DataFrame':
        """
        Transforms the DataFrame from wide format to long format.
        
        :param id_vars: Columns to keep fixed.
        :param value_vars: Columns to melt into 'variable' and 'value' columns.
        :return: A new DataFrame in melted (long) format.
        """
        if not all(var in self._columns for var in id_vars):
            raise ValueError("Some id_vars do not exist in the DataFrame.")
        if not all(var in self._columns for var in value_vars):
            raise ValueError("Some value_vars do not exist in the DataFrame.")
        
        melted_data = {var: [] for var in id_vars}
        melted_data["variable"] = []
        melted_data["value"] = []
        
        for i in range(len(self)):
            for value_var in value_vars:
                for id_var in id_vars:
                    melted_data[id_var].append(self._columns[id_var][i])
                melted_data["variable"].append(value_var)
                melted_data["value"].append(self._columns[value_var][i])
        
        new_columns = {
            name: Column(melted_data[name], Type.String if name == "variable" else Type.Float)
            for name in melted_data
        }
        
        return DataFrame(new_columns)
    def shift(self, periods: int) -> 'DataFrame':
        """
        Shifts the rows of the DataFrame by the specified number of periods.
        
        :param periods: Number of periods to shift the rows. Positive shifts down, negative shifts up.
        :return: A new DataFrame with shifted rows.
        """
        new_columns = {}
        num_rows = len(self)
        
        for name, column in self._columns.items():
            if periods > 0:
                # Posun dolů
                new_data = [None] * periods + column[:-periods] if periods < num_rows else [None] * num_rows
            elif periods < 0:
                # Posun nahoru
                new_data = column[-periods:] + [None] * (-periods) if -periods < num_rows else [None] * num_rows
            else:
                # Žádný posun
                new_data = column
            
            new_columns[name] = Column(new_data, column.dtype)
        
        return DataFrame(new_columns)
    def interpolate(self, axis: int = 0) -> 'DataFrame':
        """
        Interpolates missing values (NaN represented as None) using linear interpolation.
        
        :param axis: 0 to interpolate along columns (down each column), 1 to interpolate along rows.
        :return: A new DataFrame with interpolated values.
        """
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")
        
        def linear_interpolate(data):
            """Helper function to perform linear interpolation on a list."""
            new_data = data[:]
            for i in range(len(new_data)):
                if new_data[i] is None:
                    # Find the previous and next valid values
                    prev_index = next((j for j in range(i - 1, -1, -1) if new_data[j] is not None), None)
                    next_index = next((j for j in range(i + 1, len(new_data)) if new_data[j] is not None), None)
                    
                    if prev_index is not None and next_index is not None:
                        # Linearly interpolate
                        new_data[i] = new_data[prev_index] + (new_data[next_index] - new_data[prev_index]) * (i - prev_index) / (next_index - prev_index)
                    elif prev_index is not None:
                        # Carry forward the last valid value
                        new_data[i] = new_data[prev_index]
                    elif next_index is not None:
                        # Carry backward the next valid value
                        new_data[i] = new_data[next_index]
            return new_data
    
        new_columns = {}
    
        if axis == 0:  # Interpolate down the columns
            for name, column in self._columns.items():
                new_columns[name] = Column(linear_interpolate(column), column.dtype)
        
        elif axis == 1:  # Interpolate across the rows
            for i in range(len(self)):
                row = [self._columns[name][i] for name in self._columns]
                interpolated_row = linear_interpolate(row)
                for j, name in enumerate(self._columns):
                    if name not in new_columns:
                        new_columns[name] = []
                    new_columns[name].append(interpolated_row[j])
            
            for name in new_columns:
                new_columns[name] = Column(new_columns[name], self._columns[name].dtype)
        
        return DataFrame(new_columns)
    def drop(self, labels: list, axis: int) -> 'DataFrame':
        """
        Drops the specified rows or columns from the DataFrame.
        
        :param labels: List of row indices (if axis=0) or column names (if axis=1) to drop.
        :param axis: 0 to drop rows, 1 to drop columns.
        :return: A new DataFrame with the specified rows or columns removed.
        """
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")
        
        new_columns = {}
        
        if axis == 0:  # Drop rows
         for name, column in self._columns.items():
            # Odstraníme řádky, které odpovídají indexům v `labels`
            new_data = [item for idx, item in enumerate(column) if idx in labels]
            new_columns[name] = Column(new_data, column.dtype)
        
        elif axis == 1:  # Drop columns
            new_columns = {name: column for name, column in self._columns.items() if name not in labels}
        
        return DataFrame(new_columns)
    def dot(self, other: 'DataFrame') -> 'DataFrame':
        """
        Computes the matrix product of the DataFrame with another DataFrame.
        
        :param other: The other DataFrame to multiply with.
        :return: A new DataFrame that is the result of the matrix multiplication.
        """
        # Zkontrolujeme, zda počet sloupců v prvním DataFrame odpovídá počtu řádků ve druhém DataFrame
        if len(self._columns) != len(other._columns[list(other._columns.keys())[0]]):
            raise ValueError("Number of columns in the first DataFrame must equal the number of rows in the second DataFrame.")
        
        # Vytvoříme slovník pro nové sloupce výsledného DataFrame
        result_columns = {col_name: [] for col_name in other._columns}
        
        # Iterujeme přes každý řádek prvního DataFrame
        for i in range(len(self)):
            # Iterujeme přes každý sloupec druhého DataFrame
            for col_name in other._columns:
                result_value = 0
                # Provedeme součin odpovídajících prvků a jejich součet
                for j, self_col_name in enumerate(self._columns):
                    result_value += self._columns[self_col_name][i] * other._columns[col_name][j]
                result_columns[col_name].append(result_value)
        
        # Vytvoříme nový DataFrame z výsledných sloupců
        new_columns = {name: Column(values, Type.Float) for name, values in result_columns.items()}
        
        return DataFrame(new_columns)
    def diff(self, axis: int = 0) -> 'DataFrame':
        """
        Calculates the difference between the current and the previous row or column in the DataFrame.
        
        :param axis: 0 to calculate differences between rows, 1 to calculate differences between columns.
        :return: A new DataFrame with the differences.
        """
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")
        
        new_columns = {}
        
        if axis == 0:  # Difference between rows
            for name, column in self._columns.items():
                new_data = [None]  # First row will be NaN (None)
                for i in range(1, len(column)):
                    if column[i] is None or column[i-1] is None:
                        new_data.append(None)
                    else:
                        new_data.append(column[i] - column[i-1])
                new_columns[name] = Column(new_data, column.dtype)
        
        elif axis == 1:  # Difference between columns
            for i in range(len(self)):
                row_diff = [None]  # First column will be NaN (None)
                prev_value = None
                for j, name in enumerate(self._columns):
                    if j == 0:
                        prev_value = self._columns[name][i]
                        continue
                    
                    current_value = self._columns[name][i]
                    if current_value is None or prev_value is None:
                        row_diff.append(None)
                    else:
                        row_diff.append(current_value - prev_value)
                    
                    prev_value = current_value
                
                for k, name in enumerate(self._columns):
                    if name not in new_columns:
                        new_columns[name] = []
                    new_columns[name].append(row_diff[k])
            
            for name in new_columns:
                new_columns[name] = Column(new_columns[name], self._columns[name].dtype)
        
        return DataFrame(new_columns)

    def cumprod(self, axis: int = 0) -> 'DataFrame':
        """
        Computes the cumulative product of elements along a given axis.
        
        :param axis: 0 to compute along rows (top to bottom), 1 to compute along columns (left to right).
        :return: A new DataFrame with cumulative products.
        """
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")
        
        new_columns = {}
        
        if axis == 0:  # Cumulative product down the rows
            for name, column in self._columns.items():
                new_data = []
                cum_prod = 1
                for value in column:
                    if value is None:
                        new_data.append(None)
                    else:
                        cum_prod *= value
                        new_data.append(cum_prod)
                new_columns[name] = Column(new_data, column.dtype)
        
        elif axis == 1:  # Cumulative product across the columns
            for i in range(len(self)):
                row_cumprod = []
                cum_prod = 1
                for name in self._columns:
                    value = self._columns[name][i]
                    if value is None:
                        row_cumprod.append(None)
                    else:
                        cum_prod *= value
                        row_cumprod.append(cum_prod)
                
                for j, name in enumerate(self._columns):
                    if name not in new_columns:
                        new_columns[name] = []
                    new_columns[name].append(row_cumprod[j])
            
            for name in new_columns:
                new_columns[name] = Column(new_columns[name], self._columns[name].dtype)
        
        return DataFrame(new_columns)
    
    def transpose(self) -> 'DataFrame':
        """
        Transposes the DataFrame, switching rows with columns.
        
        :return: A new DataFrame that is the transpose of the original.
        """
        transposed_data = {}
        # Get the number of rows and columns in the original DataFrame
        row_count = len(self._columns[list(self._columns.keys())[0]])
        col_names = list(self._columns.keys())
    
        # Iterate over each row to create the transposed columns
        for i in range(row_count):
            new_col = []
            for col_name in col_names:
                new_col.append(self._columns[col_name][i])
            transposed_data[f"row_{i}"] = Column(new_col, Type.Float)
        
        return DataFrame(transposed_data)
    def compare(self, other: 'DataFrame') -> 'DataFrame':
        """
        Compares the DataFrame with another DataFrame and returns a new DataFrame
        showing only the differing values. Where values match, NaN is placed instead.
        Columns or rows that match completely are not included in the result.
        
        :param other: The other DataFrame to compare with.
        :return: A new DataFrame with only the differing values.
        """
        if len(self._columns) != len(other._columns) or len(self._columns[list(self._columns.keys())[0]]) != len(other._columns[list(other._columns.keys())[0]]):
            raise ValueError("Both DataFrames must have the same shape.")
        
        new_columns = {}
       
        for name in self._columns:
            if name in other._columns:
                self_column = self._columns[name]
                other_column = other._columns[name]
                new_data = []
       
                for i in range(len(self_column)):
                    if self_column[i] != other_column[i]:
                        new_data.append(self_column[i])
                    else:
                        new_data.append(None)
       
                # Only add the column to the result if there is at least one difference
                if any(value is not None for value in new_data):
                    new_columns[name] = Column(new_data, self_column.dtype)
       
        return DataFrame(new_columns)
    def cumsum(self, axis: int = 0) -> 'DataFrame':
       """
       Computes the cumulative sum of elements along a given axis.
       
       :param axis: 0 to compute along rows (top to bottom), 1 to compute along columns (left to right).
       :return: A new DataFrame with cumulative sums.
       """
       if axis not in (0, 1):
           raise ValueError("Axis must be 0 (rows) or 1 (columns).")
       
       new_columns = {}
       
       if axis == 0:  # Cumulative sum down the rows
           for name, column in self._columns.items():
               new_data = []
               cum_sum = 0
               for value in column:
                   if value is None:
                       new_data.append(None)
                   else:
                       cum_sum += value
                       new_data.append(cum_sum)
               new_columns[name] = Column(new_data, column.dtype)
       
       elif axis == 1:  # Cumulative sum across the columns
           for i in range(len(self)):
               row_cumsum = []
               cum_sum = 0
               for name in self._columns:
                   value = self._columns[name][i]
                   if value is None:
                       row_cumsum.append(None)
                   else:
                       cum_sum += value
                       row_cumsum.append(cum_sum)
               
               for j, name in enumerate(self._columns):
                   if name not in new_columns:
                       new_columns[name] = []
                   new_columns[name].append(row_cumsum[j])
           
           for name in new_columns:
               new_columns[name] = Column(new_columns[name], self._columns[name].dtype)
       
       return DataFrame(new_columns)
    def _listprod(self, iterable):
        """
        Helper function for self.product(). Returns total product of an
        iterable. Ignores NaN values.
        
        :param iterable: An iterable containing numbers, possibly with NaN (None) values.
        :return: The product of all numbers in the iterable, ignoring NaN values.
        """
        product = 1
        for value in iterable:
            if value is not None and not math.isnan(value):
                product *= value
        return product
    def _listrelp(self, iterable, old, new):
        """
        Helper function for self.replace(). Acts as a search-and-replace for
        iterables. This is a procedure, not a function, and modifies the original
        iterable.
        
        :param iterable: An iterable where values will be replaced.
        :param old: The value to search for and replace.
        :param new: The value to replace the old value with.
        """
        for i in range(len(iterable)):
            if iterable[i] == old:
                iterable[i] = new
    @property
    def columns(self) -> Iterable[str]:
        """
        :return: names of columns (as iterable object)
        """
        return self._columns.keys()

    def __repr__(self) -> str:
        """
        :return: string representation of dataframe (table with aligned columns)
        """
        lines = []
        lines.append(" ".join(f"{name:12s}" for name in self.columns))
        for i in range(len(self)):
            lines.append(" ".join(self._columns[cname].get_formatted_item(i, width=12)
                                     for cname in self.columns))
        return "\n".join(lines)

    def append_column(self, col_name:str, column: Column) -> None:
        """
        Appends new column to dataframe (its name has to be unique).
        :param col_name:  name of new column
        :param column: data of new column
        """
        if col_name in self._columns:
            raise ValueError("Duplicate column name")
        self._columns[col_name] = column.copy()

    def append_row(self, row: Iterable) -> None:
        """
        Appends new row to dataframe.
        :param row: tuple of values for all columns
        """
        ...

    def filter(self, col_name:str,
               predicate: Callable[[Union[float, str]], bool]) -> 'DataFrame':
        """
        Returns new dataframe with rows which values in column `col_name` returns
        True in function `predicate`.

        :param col_name: name of tested column
        :param predicate: testing function
        :return: new dataframe
        """
        ...

    def sort(self, col_name:str, ascending=True) -> 'DataFrame':
        """
        Sort dataframe by column with `col_name` ascending or descending.
        :param col_name: name of key column
        :param ascending: direction of sorting
        :return: new dataframe
        """
        ...

    def describe(self) -> str:
        """
        similar to pandas but only with min, max and avg statistics for floats and count"
        :return: string with formatted decription
        """
        ...

    def inner_join(self, other: 'DataFrame', self_key_column: str,
                   other_key_column: str) -> 'DataFrame':
        """
            Inner join between self and other dataframe with join predicate
            `self.key_column == other.key_column`.

            Possible collision of column identifiers is resolved by prefixing `_other` to
            columns from `other` data table.
        """
        ...

    def setvalue(self, col_name: str, row_index: int, value: Any) -> None:
        """
        Set new value in dataframe.
        :param col_name:  name of culumns
        :param row_index: index of row
        :param value:  new value (value is cast to type of column)
        :return:
        """
        col = self._columns[col_name]
        col[row_index] = col._cast(value)

    @staticmethod
    def read_csv(path: Union[str, Path]) -> 'DataFrame':
        """
        Read dataframe by CSV reader
        """
        return CSVReader(path).read()

    @staticmethod
    def read_json(path: Union[str, Path]) -> 'DataFrame':
        """
        Read dataframe by JSON reader
        """
        return JSONReader(path).read()


class Reader(ABC):
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)

    @abstractmethod
    def read(self) -> DataFrame:
        raise NotImplemented("Abstract method")


class JSONReader(Reader):
    """
    Factory class for creation of dataframe by CSV file. CSV file must contain
    header line with names of columns.
    The type of columns should be inferred from types of their values (columns which
    contains only value has to be floats columns otherwise string columns),
    """
    def read(self) -> DataFrame:
        with open(self.path, "rt") as f:
            json = load(f)
        columns = {}
        for cname in json.keys(): # cyklus přes sloupce (= atributy JSON objektu)
            dtype = Type.Float if all(value is None or isinstance(value, Real)
                                      for value in json[cname]) else Type.String
            columns[cname] = Column(json[cname], dtype)
        return DataFrame(columns)


class CSVReader(Reader):
    """
    Factory class for creation of dataframe by JSON file. JSON file must contain
    one object with attributes which array values represents columns.
    The type of columns are inferred from types of their values (columns which
    contains only value is floats columns otherwise string columns),
    """
    def read(self) -> 'DataFrame':
        ...


#if __name__ == "__main__":
#    df = DataFrame(dict(
#        a=Column([None, 3.1415], Type.Float),
#        b=Column(["a", 2], Type.String),
#        c=Column(range(2), Type.Float)
#        ))
#    df.setvalue("a", 1, 42)
#    print(df)
#
#    df = DataFrame.read_json("data.json")
#    print(df)
#
#for line in df:
#    print(line)
#columns = {
#    "Name": Column(["Alice", "Bob", "Alice", "Charlie", "Bob"], Type.String),
#    "Age": Column([25, 30, 25, 35, 30], Type.Float),
#    "City": Column(["New York", "Los Angeles", "New York", "Chicago", "Los Angeles"], Type.String)
#}
#
#df = DataFrame(columns)
#dfu = df.sample(2)
#print(dfu)
columns = {
    "A": Column([1, 2, 3, 4], Type.Float),
    "B": Column([5, 6, 7, 8], Type.Float),
    "C": Column([9, 10, 11, 12], Type.Float)
}

df = DataFrame(columns)
print(df)

###
df3 = df.cumsum(0)
print(df3)