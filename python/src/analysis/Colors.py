colors = {
    "red": ["#9d0006", "#cc241d", "#fb4934"],
    "purple": ["#8f3f71", "#b16286", "#d3869b"],
    "blue": ["#076678", "#458588", "#83a598"],
    "aqua": ["#427b58", "#689d6a", "#8ec07c"],
    "green": ["#79740e", "#98971a", "#b8bb26"],
    "yellow": ["#b57614", "#d79921", "#fabd2f"],
    "orange": ["#af3a03", "#d65d0e", "#fe8019"]
}


def get_colors(length: int) -> list:
    """Returns a list of colors, which contrast with their neighbours on the list.
    Args:
        length: length of the returned list
    Returns:
        List of hex values for contrasting colors
    """
    color_list = []
    keys = list(colors.keys())
    number_of_colors = len(keys)
    incr = int(number_of_colors / 2 + 1)
    for i in range(length):
        wheel_idx = incr * i % number_of_colors
        color_list.append(colors[keys[wheel_idx]][i % 3])
    return color_list

