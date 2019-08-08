/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

#include <type_traits>

//#############################################################################
template<
    typename T>
class KernelFuntionObjectTemplate
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<std::int32_t, T>::value,
            "Incorrect additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelFuntionObjectTemplate<std::int32_t> kernel;

    REQUIRE(fixture(kernel));
}
};

//#############################################################################
class KernelInvocationWithAdditionalTemplate
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        T const &) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<std::int32_t, T>::value,
            "Incorrect additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
struct TestTemplateExtra
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelInvocationWithAdditionalTemplate kernel;

    REQUIRE(fixture(kernel, std::int32_t()));
}
};

TEST_CASE( "kernelFuntionObjectTemplate", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}

TEST_CASE( "kernelFuntionObjectExtraTemplate", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateExtra() );
}
