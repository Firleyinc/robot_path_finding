import datetime
import time
def main():
    start = datetime.datetime.now()
    print("hello!")
    time.sleep(0.01)
    end = datetime.datetime.now()
    print((end-start))

if __name__ == '__main__':
    main()
