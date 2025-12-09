# xx:hh 时间转化为距离 0 点的分钟数
def time_to_minutes(time_str) :
    hours , minutes = map(int , time_str.split(":"))
    return hours * 60 + minutes
