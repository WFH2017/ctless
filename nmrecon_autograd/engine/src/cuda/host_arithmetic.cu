
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <cassert>
#include "common/config.h"
#include "cuda/host_arithmetic.h"
#include "cuda/functor.h"

NMRECON_NAMESPACE_BEGIN
namespace common
{
    namespace host_arithmetic
    {

        // 实现所有模板函数
        template <typename T>
        thrust::host_vector<T> operator+(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2)
        {
            assert(vec.size() == vec2.size());
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), thrust::placeholders::_1 + thrust::placeholders::_2);
            return vec_out;
        }

        template <typename T, typename T2>
        thrust::host_vector<T> operator+(const thrust::host_vector<T> &vec, const T2 &val)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), thrust::placeholders::_1 + val);
            return vec_out;
        }

        template <typename T, typename T2>
        thrust::host_vector<T> operator+(const T2 &val, const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), thrust::placeholders::_1 + val);
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> operator-(const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), -thrust::placeholders::_1);
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> operator-(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2)
        {
            assert(vec.size() == vec2.size());
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), thrust::placeholders::_1 - thrust::placeholders::_2);
            return vec_out;
        }

        template <typename T, typename T2>
        thrust::host_vector<T> operator-(const thrust::host_vector<T> &vec, const T2 &val)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), thrust::placeholders::_1 - val);
            return vec_out;
        }

        template <typename T, typename T2>
        thrust::host_vector<T> operator-(const T2 &val, const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), val - thrust::placeholders::_1);
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> operator*(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2)
        {
            assert(vec.size() == vec2.size());
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), thrust::placeholders::_1 * thrust::placeholders::_2);
            return vec_out;
        }

        template <typename T, typename T2>
        thrust::host_vector<T> operator*(const thrust::host_vector<T> &vec, const T2 &val)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), thrust::placeholders::_1 * val);
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> operator/(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2)
        {
            assert(vec.size() == vec2.size());
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), thrust::placeholders::_1 / thrust::placeholders::_2);
            return vec_out;
        }

        template <typename T, typename T2>
        thrust::host_vector<T> operator/(const thrust::host_vector<T> &vec, const T2 &val)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), thrust::placeholders::_1 / val);
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> true_div(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &val)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), val.begin(), vec_out.begin(), functor::true_divide_functor<T>());
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> div_with_threshold(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &val, const T threshold, const T _default)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), val.begin(), vec_out.begin(), functor::divide_thres_functor<T>(threshold, _default));
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> absolute(const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), functor::abs_functor<T>());
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> pow(const thrust::host_vector<T> &vec, const float val)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), functor::pow_functor<T>(val));
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> sqrt(const thrust::host_vector<T> &vec)
        {
            return pow<T>(vec, 0.5);
        }

        template <typename T>
        thrust::host_vector<T> exp(const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), functor::exp_functor<T>());
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> log(const thrust::host_vector<T> &vec, const T base)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), functor::log_functor<T>(base));
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> remove_nan(const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), functor::remove_nan_functor<T>());
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> remove_inf(const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), functor::remove_inf_functor<T>());
            return vec_out;
        }

        template <typename T>
        thrust::host_vector<T> remove_negative(const thrust::host_vector<T> &vec)
        {
            thrust::host_vector<T> vec_out(vec.size());
            thrust::transform(vec.begin(), vec.end(), vec_out.begin(), functor::remove_negative_functor<T>());
            return vec_out;
        }

        template <typename TInd, typename T>
        thrust::host_vector<T> search_lookup(const thrust::host_vector<TInd> &index, const thrust::host_vector<T> &lookup_table)
        {
            auto iter = thrust::make_permutation_iterator(lookup_table.begin(), index.begin());
            return thrust::host_vector<T>(iter, iter + index.size());
        }

        template <typename T1, typename T2>
        thrust::host_vector<T2> convert(const thrust::host_vector<T1> &data)
        {
            return thrust::host_vector<T2>(data.begin(), data.end());
        }

    } // end namespace host_arithmetic
} // end namespace common
NMRECON_NAMESPACE_END

// 实例化常用类型
// 显式实例化所有 host_arithmetic 的模板函数 float 类型
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator+(const thrust::host_vector<float> &, const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator+(const thrust::host_vector<float> &, const float &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator+(const float &, const thrust::host_vector<float> &);

template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator-(const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator-(const thrust::host_vector<float> &, const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator-(const thrust::host_vector<float> &, const float &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator-(const float &, const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator*(const thrust::host_vector<float> &, const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator*(const thrust::host_vector<float> &, const float &);

template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator/(const thrust::host_vector<float> &, const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::operator/(const thrust::host_vector<float> &, const float &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::true_div(const thrust::host_vector<float> &, const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::div_with_threshold(const thrust::host_vector<float> &, const thrust::host_vector<float> &, const float threshold, const float _default);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::absolute(const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::pow(const thrust::host_vector<float> &, const float val);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::sqrt(const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::exp(const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::log(const thrust::host_vector<float> &, const float base);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::remove_nan(const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::remove_inf(const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::remove_negative(const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::search_lookup(const thrust::host_vector<int> &, const thrust::host_vector<float> &);
template thrust::host_vector<float> nmrecon::common::host_arithmetic::convert<int, float>(const thrust::host_vector<int> &);
