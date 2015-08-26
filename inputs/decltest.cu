void foo(int i){
	int a;
	a = i;
	foo(a);
}

int main(){
	foo(0);
}
