def main():
    print("Wellcome to email Slicer " + "\n")
    print("")

    email_input = input("Enter your Email Address:  ")

    (username, domain) = email_input.split("@")

    (domain, extension)= domain.split(".")

    print("Username : " , username)
    print("Domain : " , domain)
    print("Extension : ", extension)

while True:
    main()