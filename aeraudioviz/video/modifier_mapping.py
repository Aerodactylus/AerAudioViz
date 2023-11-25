from typing import Union


class ModifierMapping:

    def __init__(self, modifier_function, modifier_column: str,
                 **kwarg_ranges: tuple[Union[float, int], Union[float, int]]):
        """

        :param modifier_function: a method from the ImageModifiers class
        :param modifier_column: the column in the feature DataFrame that will control the image modificaitons
        :param kwarg_ranges: Key ward arguments.
        Key is the key word argument for the modifier_function that the feature data will control value of.
        Value is a tuple containing min and max range of the key word argument value to pass to modifier_function.
        """
        self.modifier_function = modifier_function
        self.modifier_column = modifier_column
        self.kwarg_ranges = kwarg_ranges
