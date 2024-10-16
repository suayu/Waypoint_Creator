import runtime
import PurePursuit
if __name__ == '__main__':
    #simple_test = PurePursuit.SimpleTest()
    path = PurePursuit.SimpleTest.simple_read('./res/res_0.txt')

    test = PurePursuit.SimpleTest()
    test.offline_test(path,120)
