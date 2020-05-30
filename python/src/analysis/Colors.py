color_wheel = {
    "yellow": ['#a19600', '#cbbe00', '#fef200', '#fbf583', '#fefaca'],
    "yellow_orange": ['#9a6603', '#c18210', '#f8a51b', '#fcc777', '#ffe3cb'],
    "orange": ['#994f06', '#c26919', '#f58225', '#f5a872', '#fbdcc8'],
    "red_orange": ['#972d1d', '#bb302d', '#ed403c', '#f58d72', '#fdcfc0'],
    "red": ['#8f0407', '#b71018', '#ee1c25', '#f47a6d', '#fad2d0'],
    "red_purple": ['#69015a', '#810d70', '#a3218e', '#bd7cb4', '#e0cde3'],
    "purple": ['#38085c', '#491d76', '#592f93', '#7e6aaf', '#b8add5'],
    "blue_purple": ['#081c63', '#172c7c', '#21409a', '#5664af', '#8b92c6'],
    "blue": ['#013d71', '#004e8c', '#0465b2', '#5c88c5', '#aec5e4'],
    "blue_green": ['#006e69', '#018989', '#00acac', '#56c4c5', '#bce4e6'],
    "green": ['#006c3b', '#00864b', '#03a45e', '#64c195', '#c0dfcd'],
    "yellow_green": ['#3f7829', '#569834', '#70be44', '#add67a', '#e1edd7'],
}


def get_colors(length: int, shade: int) -> list:
    """Returns a list of colors, which contrast with their neighbours on the list.
    Returns lighter shades after number of colors in the wheel and repeats after
    twice the number.
    Args:
        length: length of the returned list
        shade: shade of the returned colors, 0 for darker, 1 for lighter
    Returns:
        List of hex values for contrasting colors
    """
    dark_shade = shade
    light_shade = shade + 2
    color_list = []
    keys = list(color_wheel.keys())
    number_of_colors = len(keys)
    incr = int(number_of_colors / 2 + 1)
    for i in range(length):
        wheel_idx = incr * i % number_of_colors
        color_shade = dark_shade if int(i / number_of_colors) % 2 == 0 else light_shade
        color_list.append(color_wheel[keys[wheel_idx]][color_shade])
    return color_list

