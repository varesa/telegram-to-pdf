from image import fix_image


def main():
    for i in range(1, 7):
        path = f'/home/esav.fi/esa/Pictures/cvtest/{i}.jpg'
        fix_image(path)


if __name__ == '__main__':
    main()

