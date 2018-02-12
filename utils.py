def get_time_diff(start,end):
    temp = end - start
    hours = temp // 3600
    temp = temp - 3600 * hours
    minutes = temp // 60
    seconds = temp - 60 * minutes
    diff='%d:%d:%d' % (hours, minutes, seconds)
    return diff