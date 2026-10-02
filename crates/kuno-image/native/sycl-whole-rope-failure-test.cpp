// CPU-only fixed numerical failure versus uncertain boundary/cleanup fixtures.
#define KUNO_WHOLE_ROPE_FAILURE_TEST
#define main whole_rope_test_entry
#include "sycl-whole-rope-test.cpp"
#undef main
int main(int argc,char** argv) {
    if(argc!=2||std::strlen(argv[1])!=1||argv[1][0]<'1'||argv[1][0]>'9')return 2;
    injected_failure=unsigned(argv[1][0]-'0');
    if(injected_failure>=7) {
        struct Witness{~Witness(){std::puts("whole_rope_failure_unwind");}} witness;
        std::vector<float> host_dma_witness(64,1.f);
        device_mode=true;
        if(injected_failure==7)checked([]{throw 1;});
        else if(injected_failure==8)status_checked(GGML_STATUS_FAILED,9);
        else backend_result(false,7);
        (void)witness;(void)host_dma_witness;return 1;
    }
    char* args[]={argv[0],nullptr};
    const int result=whole_rope_test_entry(1,args);
    bool valid=result==1;
    if(injected_failure==1) {
        valid&=!uncertain;
        for(unsigned i=1;i<=4;++i)valid&=released[i]==1&&release_attempts[i]==1;
    } else {
        valid&=uncertain;
        if(injected_failure==5) {
            for(unsigned i=1;i<=3;++i)valid&=released[i]==158&&release_attempts[i]==158;
            valid&=released[4]==0&&release_attempts[4]==1;
        } else {
            for(unsigned i=1;i<=4;++i)valid&=released[i]==0;
            if(injected_failure==4)valid&=release_attempts[1]==1;
            else for(unsigned i=1;i<=4;++i)valid&=release_attempts[i]==0;
        }
    }
    if(!valid){std::printf("whole_rope_failure_error fixture=%u\n",injected_failure);return 1;}
    std::printf("whole_rope_failure_completed fixture=%u fixed_exit=1 no_retry=1\n",injected_failure);return 0;
}
