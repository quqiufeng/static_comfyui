def main():
    a: list[float] = make_float_array(640)
    i: int = 0
    while i < 640:
        float_array_set(a, i, float(i))
        i = i + 1
    print("a[0]="); print(float_array_ref(a,0))
    print("a[1]="); print(float_array_ref(a,1))
    print("a[2]="); print(float_array_ref(a,2))
    print("a[100]="); print(float_array_ref(a,100))
    print("a[400]="); print(float_array_ref(a,400))
    print("a[639]="); print(float_array_ref(a,639))
    print("sum="); print(arr_sum(a,640))
    exit_program(0)
main()
