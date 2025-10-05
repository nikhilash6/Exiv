
# ------- Simple lambdas
nearest_multiple = lambda x, n: n * (x // n)

def clamp(x, min_value = None, max_value = None):
    if min_value is not None and min_value >= x:
        x = min_value
    
    if max_value is not None and max_value <= x:
        x = max_value
        
    return x